# Expects artifacts per quantized layer: { "q", "scale", "bias"(optional), "meta": {"packed", "shape", "group_size"(optional)} }
# Works with packed int4 (preferred) or unpacked int8.
# Supports grouped quantization (group_size > 1) and per-channel quantization (group_size = 1).
# Tries to use AWQ fused kernel (awq_inference_engine) if available, else falls back to dequant->FP16 matmul.
# Provides validation helpers.

import os
import json
from copy import deepcopy
import torch
import torch.nn as nn
import math
import importlib
from typing import Dict, Tuple, Any, Optional, List
import torch.nn.functional as F


# Helper: int4 unpack
def unpack_int4(packed: torch.ByteTensor, out_shape: Tuple[int, int], device: Optional[torch.device] = None) -> torch.IntTensor:
    """
    Unpack packed int4 bytes into signed int8/int32 Tensor with shape out_shape (out_features, in_features).
    packed: ByteTensor of length ceil(prod(out_shape)/2)
    out_shape: (out, in)
    Returns int8 tensor in range [-8, 7].
    """
    if not packed.dtype == torch.uint8 and not packed.dtype == torch.int8:
        packed = packed.to(torch.uint8)
    flat = packed.view(-1).to(torch.uint8)
    # low nibble:
    lo = (flat & 0x0F).to(torch.int8)
    hi = (flat >> 4 & 0x0F).to(torch.int8)

    # convert two's complement 4-bit to signed:
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)

    # interleave lo, hi -> sequence length 2*N
    inter = torch.empty(flat.size(0) * 2, dtype=torch.int8, device=device if device else flat.device)
    inter[0::2] = lo
    inter[1::2] = hi

    total_needed = out_shape[0] * out_shape[1]
    inter = inter[:total_needed]
    return inter.view(out_shape).to(torch.int8)


class QuantLinear(nn.Module):
    """
    Quantized linear wrapper for baked DACQ / AWQ artifacts.
    artifact: dict with keys:
      - "q": packed bytes (torch.uint8) if meta['packed']==True, else signed int8 tensor (out,in)
      - "scale": float tensor (shape [out] or [out/groups]) representing Δ''_c
      - "bias": optional float tensor (out,)
      - "meta": { "packed": bool, "shape": (out,in), "group_size": int (optional) }
    """
    def __init__(self, artifact: Dict[str, Any], allow_fused: bool = True):
        super().__init__()
        self.artifact = artifact
        self.meta = artifact.get("meta", {})
        # Validate meta
        assert "shape" in self.meta, "artifact meta must include 'shape'=(out,in)"
        self.out_shape = tuple(self.meta["shape"])
        assert len(self.out_shape) == 2, "meta.shape must be (out, in)"
        self.group_size = int(self.meta.get("group_size", 1))
        assert isinstance(self.group_size, int) and self.group_size >= 1
        self.packed = bool(self.meta.get("packed", False))

        # load scale and bias
        scale = artifact["scale"]
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        # scale shape sanity: if grouped, scale length should be out_features / group_size
        expected_scale_len = self.out_shape[0] // self.group_size
        if scale.numel() not in (1, expected_scale_len, self.out_shape[0]):
            raise ValueError(f"Scale length {scale.numel()} incompatible with out={self.out_shape[0]} and group_size={self.group_size}")
        self.register_buffer("scale", scale.contiguous().float())

        bias = artifact.get("bias", None)
        if bias is not None:
            if not isinstance(bias, torch.Tensor):
                bias = torch.tensor(bias)
            # store bias in FP16 on CUDA else FP32
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.register_buffer("bias", bias.to(dtype))
        else:
            self.bias = None

        # store q
        if self.packed:
            q_bytes = artifact["q"]
            if not isinstance(q_bytes, torch.Tensor):
                q_bytes = torch.tensor(q_bytes, dtype=torch.uint8)
            self.register_buffer("q_packed", q_bytes.to(torch.uint8))
            self.q_int = None
        else:
            q_int = artifact["q"]
            if not isinstance(q_int, torch.Tensor):
                q_int = torch.tensor(q_int, dtype=torch.int8)
            # ensure shape matches out_shape
            if tuple(q_int.shape) != self.out_shape:
                raise ValueError(f"Unpacked q shape {tuple(q_int.shape)} does not match meta.shape {self.out_shape}")
            self.register_buffer("q_int", q_int.to(torch.int8))
            self.q_packed = None

        # Attempt to load fused CUDA extension if present
        self.fused = None
        if allow_fused:
            try:
                self.fused = importlib.import_module("awq_inference_engine")
            except Exception:
                self.fused = None  # not available

    def _expand_scale_for_matmul(self, device):
        """
        Returns scale expanded to (out_features, 1) for broadcasting with q_int (out,in).
        Handles grouped scales.
        """
        s = self.scale.to(device)
        if s.numel() == 1:
            return s.view(1, 1)  # scalar
        if self.group_size == 1:
            # per-output-channel
            return s.view(-1, 1)
        else:
            # grouped: s length = out_features / group_size
            g = self.group_size
            out = self.out_shape[0]
            # repeat each group-size times
            s_expanded = s.repeat_interleave(g).view(out, 1)
            return s_expanded.to(device)

    def _get_q_int(self, device):
        """
        Return q_int (out, in) as int8 on device.
        If packed and fused kernel supports packed input, we won't call this in forward.
        """
        if self.q_int is not None:
            return self.q_int.to(device)
        assert self.q_packed is not None, "No q present"
        return unpack_int4(self.q_packed, out_shape=self.out_shape, device=device).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features) or (seq, in_features)
        returns: (batch, out_features)
        """
        # Prefer fused kernel if available AND supports packed int4
        if self.fused is not None:
            try:
                # Many awq kernels expose a matmul_int4-like API:
                # Call signatures vary across releases; we attempt common ones.
                if self.packed:
                    # If kernel supports packed int4 directly:
                    # commonly: fused.matmul_int4(input, q_packed, scale, bias=None)
                    if x.is_cuda:
                        out = self.fused.matmul_int4(x, self.q_packed, self.scale, bias=self.bias if self.bias is None else self.bias)
                    else:
                        # move inputs to cuda for fused kernel
                        x_cuda = x.cuda()
                        out_cuda = self.fused.matmul_int4(x_cuda, self.q_packed.cuda(), self.scale.cuda(), bias=self.bias.cuda() if self.bias is not None else None)
                        out = out_cuda.cpu()
                    return out
                else:
                    # if q already unpacked int8:
                    if x.is_cuda:
                        out = self.fused.matmul_int8(x, self.q_int, self.scale, bias=self.bias if self.bias is None else self.bias)
                    else:
                        x_cuda = x.cuda()
                        out_cuda = self.fused.matmul_int8(x_cuda, self.q_int.cuda(), self.scale.cuda(), bias=self.bias.cuda() if self.bias is not None else None)
                        out = out_cuda.cpu()
                    return out
            except Exception:
                # fused kernel present but API mismatch -> fall back safely
                pass

        # Fallback path: dequantize to FP16/BF16 and use native matmul
        device = x.device
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        q_int = self._get_q_int(device=device).to(torch.int8)  # (out, in)
        scale_mat = self._expand_scale_for_matmul(device=device)  # (out,1)
        # broadcast multiply
        W_fp = (scale_mat.float() * q_int.float()).to(dtype)  # (out, in)
        out = x.matmul(W_fp.t())
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


# Model replacement utilities

def replace_linears_with_quant(model: nn.Module, artifacts_map: Dict[str, Dict[str, Any]], allow_fused: bool = True) -> int:
    """
    Recursively replace nn.Linear (and optionally named custom linear modules) with QuantLinear when an artifact is available.
    artifacts_map keys should match module paths (e.g. "transformer.h.3.mlp.fc1").
    Handles tied weights by using same artifact object for both references if provided.
    Returns number of replaced modules.
    """
    replaced = 0
    for name, child in list(model.named_children()):
        full_name = name if not hasattr(model, "_name_prefix") else f"{model._name_prefix}.{name}"  # fallback
        # Better approach: compute module path by walking outer function (we accept artifact keys matching huggingface names)
        # We'll build full_name properly by using model.named_modules in the caller context for deterministic mapping.
        if isinstance(child, nn.Linear):
            # try direct mapping using layer path found via recursion where prefixing happens externally
            # In practice, user should pass artifacts_map keyed by names from model.named_modules()
            # We'll look up by trying a few forms:
            # 1) exact name
            # 2) prefix.name not available here — so skip; caller should use replace_by_names helper
            pass
        else:
            replaced += replace_linears_with_quant(child, artifacts_map, allow_fused)
    return replaced

def replace_by_names(model: nn.Module, artifacts_map: Dict[str, Dict[str, Any]], allow_fused: bool = True) -> int:
    """
    Replace modules matching keys in artifacts_map.
    Keys must be module paths exactly as returned by model.named_modules().
    Example key: "transformer.h.3.mlp.fc1"
    """
    # Build mapping from name -> module parent and attr name so we can replace
    name_to_parent = {}
    for full_name, module in model.named_modules():
        # split into parent path and attr
        if "." in full_name:
            parent_path, attr = full_name.rsplit(".", 1)
        else:
            parent_path, attr = "", full_name
        name_to_parent[full_name] = (parent_path, attr)

    replaced = 0
    for key, artifact in artifacts_map.items():
        if key not in name_to_parent:
            # try direct attr on top-level
            continue
        parent_path, attr = name_to_parent[key]
        # navigate to parent
        parent = model
        if parent_path != "":
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        # ensure module exists
        old_mod = getattr(parent, attr)
        # create quant module
        qmod = QuantLinear(artifact, allow_fused=allow_fused)
        setattr(parent, attr, qmod)
        replaced += 1
    return replaced


# Activation capture / validation helpers

def capture_activations(model: nn.Module, inputs: Dict[str, torch.Tensor], device: torch.device, module_filter=None) -> Dict[str, torch.Tensor]:
    """
    Run a forward pass and capture outputs of modules matching module_filter.
    module_filter: callable(module_name, module) -> bool. If None, captures Linear and QuantLinear outputs.
    Returns dict name->activation (FP32, CPU)
    """
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().float().cpu()
        return hook

    if module_filter is None:
        def module_filter(name, m):
            return isinstance(m, (nn.Linear, QuantLinear))

    for name, module in model.named_modules():
        if module_filter(name, module):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model.to(device)
        _ = model(**inputs)

    for h in hooks: h.remove()
    return activations


def logits_mse_and_topk_overlap(model_fp: nn.Module, model_q: nn.Module, tokenizer, prompts: List[str], device: torch.device, k: int = 5):
    """
    Run models on prompts, return logits MSE, max abs diff, and top-k overlap %
    """
    model_fp.eval()
    model_q.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out_fp = model_fp(**enc, return_dict=True)
        out_q = model_q(**enc, return_dict=True)
        logits_fp = out_fp.logits.detach().float()
        logits_q = out_q.logits.detach().float()
    mse = torch.mean((logits_fp - logits_q).pow(2)).item()
    max_abs = torch.max(torch.abs(logits_fp - logits_q)).item()

    overlaps = []
    for i in range(len(prompts)):
        last_fp = F.softmax(logits_fp[i, -1, :], dim=-1)
        last_q = F.softmax(logits_q[i, -1, :], dim=-1)
        topk_fp = torch.topk(last_fp, k).indices.cpu().tolist()
        topk_q = torch.topk(last_q, k).indices.cpu().tolist()
        overlap = len(set(topk_fp) & set(topk_q))
        overlaps.append(overlap / k)
    avg_overlap = sum(overlaps) / len(overlaps)
    return {"mse": mse, "max_abs": max_abs, "topk_overlap": avg_overlap}


def perplexity_on_texts(model: nn.Module, tokenizer, texts: List[str], device: torch.device) -> float:
    """
    Compute perplexity (exponential of average cross-entropy) on a small list of texts.
    Model must return .loss when labels provided (huggingface causal models do).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt").to(device)
            out = model(**enc, labels=enc["input_ids"])
            loss = out.loss.item()
            n = enc["input_ids"].numel()
            total_loss += loss * n
            total_tokens += n
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# Helpers for usage

def artifact_name_to_module_name(fname: str) -> str:
    """
    Convert an artifact filename to the model module path you used for quantization.
    Example: 'transformer.h.3.mlp.fc1.pt' -> 'transformer.h.3.mlp.fc1'
    Adjust this to your project's naming convention.
    """
    return fname.replace(".pt", "")

def load_artifacts_from_dir(artifacts_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all .pt artifact files in artifacts_dir and return a mapping
    of module_name -> artifact_dict expected by QuantLinear.
    """
    mapping = {}
    for fname in os.listdir(artifacts_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(artifacts_dir, fname)
        obj = torch.load(path, map_location="cpu")
        # Basic validation of keys
        if not all(k in obj for k in ("q", "scale", "meta")):
            print(f"Skipping {fname}: missing required keys (q, scale, meta)")
            continue
        module_name = artifact_name_to_module_name(fname)
        mapping[module_name] = {
            "q": obj["q"],
            "scale": obj["scale"],
            "bias": obj.get("bias", None),
            "meta": obj["meta"]
        }
    return mapping


def main():
    # -----------------------
    # User configurable
    # -----------------------
    MODEL_ID = "meta-llama/Llama-3-8b"   # replace with local path for LLaMA / Qwen
    ARTIFACTS_DIR = "/path/to/dacq_artifacts" # directory with per-layer .pt artifacts
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROMPTS = ["The capital of France is", "Machine learning models are"]
    SMALL_DEV = ["This is a short test sentence for perplexity.", "Another short evaluation sentence."]

    # -----------------------
    # Load HF model & tokenizer
    # -----------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading tokenizer and model (may take a while)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model_fp16.eval()

    # Make a deepcopy for quantized model (we'll replace some modules)
    print("Cloning model for quantized replacement (this keeps FP copy intact)...")
    model_q = deepcopy(model_fp16)

    # -----------------------
    # Load quant artifacts
    # -----------------------
    print("Loading artifacts from", ARTIFACTS_DIR)
    artifacts_map = load_artifacts_from_dir(ARTIFACTS_DIR)
    if len(artifacts_map) == 0:
        raise RuntimeError("No artifacts loaded. Check ARTIFACTS_DIR and your filenames.")

    # -----------------------
    # Replace modules in model_q
    # -----------------------
    print("Replacing modules in model_q using artifacts map...")
    replaced = replace_by_names(model_q, artifacts_map, allow_fused=True)
    print(f"Replaced {replaced} modules with QuantLinear wrappers")

    # Move models to device
    model_fp16.to(DEVICE)
    model_q.to(DEVICE)

    # -----------------------
    # Run smoke forward pass
    # -----------------------
    print("Running smoke forward pass on a sample prompt...")
    enc = tokenizer(PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out_fp = model_fp16(**enc, return_dict=True)
        out_q  = model_q(**enc, return_dict=True)

    # Basic checks
    logits_fp = out_fp.logits.detach().float()
    logits_q  = out_q.logits.detach().float()
    if torch.isnan(logits_q).any() or torch.isinf(logits_q).any():
        raise RuntimeError("Quantized model produced NaN/Inf logits - check artifacts and shapes")

    print("Smoke forward OK. Computing diagnostics...")

    # -----------------------
    # Diagnostics: logits MSE + top-k overlap
    # -----------------------
    diag = logits_mse_and_topk_overlap(model_fp16, model_q, tokenizer, PROMPTS, DEVICE, k=5)
    print("Logits diagnostics:", diag)

    # -----------------------
    # Perplexity on small dev set
    # -----------------------
    try:
        ppl_fp = perplexity_on_texts(model_fp16, tokenizer, SMALL_DEV, DEVICE)
        ppl_q  = perplexity_on_texts(model_q, tokenizer, SMALL_DEV, DEVICE)
        print(f"PPL FP16: {ppl_fp:.3f}, PPL QUANT: {ppl_q:.3f}, relative increase: {(ppl_q/ppl_fp - 1)*100:.2f}%")
        diag["ppl_fp"] = ppl_fp
        diag["ppl_q"] = ppl_q
    except Exception as e:
        print("Perplexity check failed (model may not return loss automatically):", e)

    # -----------------------
    # Activation capture (optional) - run a single prompt and compare per-layer MSE
    # -----------------------
    inputs = tokenizer("Diagnostic activation capture prompt.", return_tensors="pt").to(DEVICE)
    acts_fp = capture_activations(model_fp16, inputs, DEVICE)
    acts_q  = capture_activations(model_q, inputs, DEVICE)
    layer_mse = {}
    for name, a_fp in acts_fp.items():
        if name in acts_q:
            a_q = acts_q[name]
            mse = torch.mean((a_fp - a_q).pow(2)).item()
            layer_mse[name] = mse
    # top-5 worst layers
    worst = sorted(layer_mse.items(), key=lambda x: -x[1])[:5]
    print("Top-5 worst layers by activation MSE:", worst)
    diag["layer_mse_top5"] = worst

    # -----------------------
    # Save diagnostics report
    # -----------------------
    out_report = {"model_id": MODEL_ID, "artifacts_dir": ARTIFACTS_DIR, "device": str(DEVICE), "diagnostics": diag}
    report_path = os.path.join(ARTIFACTS_DIR, "dacq_inference_report.json")
    with open(report_path, "w") as f:
        json.dump(out_report, f, indent=2)
    print("Saved diagnostic report to", report_path)

    # -----------------------
    # Quick generation test (optional)
    # -----------------------
    try:
        prompt = "Write a short poem about autumn:"
        encg = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        print("Generating from quantized model (small sample)...")
        gen = model_q.generate(**encg, max_new_tokens=32, do_sample=False)
        print("Generated:", tokenizer.decode(gen[0], skip_special_tokens=True))
    except Exception as e:
        print("Generation test failed:", e)

if __name__ == "__main__":
    main()
