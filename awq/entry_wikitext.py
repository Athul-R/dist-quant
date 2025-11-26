#!/usr/bin/env python
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory

from awq.quantize.pre_quant import apply_awq, run_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.utils.utils import simple_dispatch_model


# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path of the HF model (local dir or hub id)",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="float16",
    choices=["float16", "bfloat16"],
    help="Compute dtype for base model / pre-quant",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Not used (eval is chunked); kept for compatibility",
)
parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Where to dump metrics JSON",
)

# Base (unquantized) evaluation
parser.add_argument(
    "--base_eval",
    action="store_true",
    help="Evaluate the base (unquantized) model without any AWQ/quantization",
)

# Model parallel / memory
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help=(
        "List of device_id:max_memory pairs to be parsed into a dictionary; "
        "Example: 0:10GiB 1:10GiB cpu:30GiB; "
        "See HF Accelerate docs for details."
    ),
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="(Optional) automatically configure parallelism with AWQ auto_parallel (not used here)",
)

# Quantization config
parser.add_argument(
    "--w_bit",
    type=int,
    default=None,
    help="Weight bitwidth for quantization (e.g., 4)",
)
parser.add_argument(
    "--q_group_size",
    type=int,
    default=-1,
    help="Group size for group-wise quantization; -1 means per-tensor",
)
parser.add_argument(
    "--no_zero_point",
    action="store_true",
    help="Disable zero_point (by default zero_point is used)",
)
parser.add_argument(
    "--q_backend",
    type=str,
    default="real",
    choices=["real", "fake"],
    help="Quantization backend. 'real' = real int weights; 'fake' = pseudo quantization",
)

# Save/load real quantized weights
parser.add_argument(
    "--dump_quant",
    type=str,
    default=None,
    help="Path to save real-quantized model state_dict",
)
parser.add_argument(
    "--load_quant",
    type=str,
    default=None,
    help="Path to load pre-computed real-quantized model state_dict",
)

# AWQ search results (scales, etc.)
parser.add_argument(
    "--run_awq",
    action="store_true",
    help="Perform AWQ search process and exit (no eval)",
)
parser.add_argument(
    "--dump_awq",
    type=str,
    default=None,
    help="Save AWQ search results here (.pt)",
)
parser.add_argument(
    "--load_awq",
    type=str,
    default=None,
    help="Load AWQ search results from here (.pt)",
)

args = parser.parse_args()

# Parse max_memory into a dict for HF Accelerate
max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}


# -------------------------
# Quantization config dict
# -------------------------
q_config = {
    "zero_point": not args.no_zero_point,
    "q_group_size": args.q_group_size,
}
print("Quantization config:", q_config)


# -------------------------
# Utilities
# -------------------------
def get_main_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def compute_model_size_gb(model: nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 3)


# -------------------------
# Build model & tokenizer
# -------------------------
def build_model_and_tokenizer(model_path: str, dtype: str):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if not os.path.exists(model_path) and not model_path.startswith(
        ("hf://", "meta-llama/", "mistralai/", "Qwen/", "deepseek")
    ):
        raise FileNotFoundError(f"{model_path} not found as a local path!")

    print(f"* Loading model from {model_path}")

    # Prepare config & tokenizer
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    # -------------------------
    # Base (unquantized) eval path
    # -------------------------
    if args.base_eval:
        print("[BaseEval] Loading base model without quantization/AWQ...")
        kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()

        kwargs_mem = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs_mem,
        )
        model = dispatch_model(model, device_map=device_map)
        return model, tokenizer

    # -------------------------
    # Real-quantized / AWQ paths
    # -------------------------
    if args.load_quant is not None:
        print(f"[Quant] Loading pre-computed real-quantized weights from: {args.load_quant}")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )
        model.tie_weights()

        kwargs_mem = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs_mem,
        )
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        model = simple_dispatch_model(model, device_map=device_map)
        model.eval()
        return model, tokenizer

    print("[Quant] Loading FP16/BF16 model as quantization source...")
    kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()

    if args.run_awq:
        assert args.dump_awq, "Please save AWQ results with --dump_awq when using --run_awq"
        print("[AWQ] Running AWQ search...")
        awq_results = run_awq(
            model,
            tokenizer,
            w_bit=args.w_bit,
            q_config=q_config,
            n_samples=128,
            seqlen=512,
        )
        os.makedirs(os.path.dirname(args.dump_awq), exist_ok=True)
        torch.save(awq_results, args.dump_awq)
        print("[AWQ] Results saved at", args.dump_awq)
        print("[AWQ] Exiting after search (no evaluation).")
        exit(0)

    if args.load_awq:
        print("[AWQ] Loading pre-computed AWQ results from", args.load_awq)
        awq_results = torch.load(args.load_awq, map_location="cpu")
        apply_awq(model, awq_results)

    if args.w_bit is not None:
        if args.q_backend == "fake":
            print(f"[Quant] Applying fake (pseudo) quantization: w_bit={args.w_bit}")
            pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
            if args.dump_quant:
                raise ValueError("Please use --q_backend real to dump quantized weights")
        elif args.q_backend == "real":
            print(f"[Quant] Applying real quantization: w_bit={args.w_bit}")
            real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)

            if args.dump_quant:
                if not args.dump_quant.endswith("v2.pt"):
                    print("[Info] Auto-change the dump_quant file name to *v2.pt")
                    args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                dirpath = os.path.dirname(args.dump_quant)
                os.makedirs(dirpath, exist_ok=True)
                print(f"[Quant] Saving quantized model state_dict at {args.dump_quant}...")
                torch.save(model.cpu().state_dict(), args.dump_quant)
                print("[Quant] Exiting after dump (no evaluation).")
                exit(0)

    kwargs_mem = {
        "max_memory": get_balanced_memory(
            model, max_memory if len(max_memory) > 0 else None
        )
    }
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "DecoderLayer",
        ],
        **kwargs_mem,
    )
    model = dispatch_model(model, device_map=device_map)

    return model, tokenizer


# -------------------------
# WikiText evaluation
# -------------------------
def load_wikitext(split: str = "test"):
    """
    Load WikiText-2 raw split.
    """
    print(f"[WikiText] Loading wikitext/wikitext-2-raw-v1, split={split} ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    print(f"[WikiText] Loaded {len(ds)} documents.")
    return ds


def evaluate_wikitext(model, tokenizer, dataset, max_length: int = 2048):
    """
    Evaluate perplexity and system metrics on WikiText-2.
    """
    model.eval()
    device = get_main_device()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Flatten dataset into a single sequence
    enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    nsamples = input_ids.numel() // max_length
    if nsamples == 0:
        raise ValueError("Not enough tokens to form a single sequence.")

    total_tokens = 0
    total_nll = 0.0

    start_time = time.perf_counter()

    for i in range(nsamples):
        batch = input_ids[:, (i * max_length) : ((i + 1) * max_length)]
        with torch.no_grad():
            logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:]

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        token_count = shift_labels.numel()
        total_tokens += token_count
        total_nll += loss.item() * token_count

        if (i + 1) % 10 == 0:
            print(
                f"[WikiText] Processed {(i + 1) * max_length} tokens | "
                f"chunks_done={i + 1}/{nsamples}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)))
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    latency_ms_per_token = 1000.0 / throughput if throughput > 0 else float("inf")

    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    else:
        peak_mem_gb = None

    model_size_gb = compute_model_size_gb(model)

    metrics = {
        "wikitext_ppl": ppl,
        "num_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "throughput_tokens_per_s": throughput,
        "latency_ms_per_token": latency_ms_per_token,
        "model_size_gb": model_size_gb,
        "peak_gpu_mem_gb": peak_mem_gb,
    }
    return metrics


# -------------------------
# Main
# -------------------------
def main():
    if args.base_eval:
        args.run_awq = False
        args.w_bit = None if args.w_bit is not None else None
        args.load_awq = None
        args.dump_awq = None
        args.load_quant = None
        args.dump_quant = None

    model, tokenizer = build_model_and_tokenizer(args.model_path, args.dtype)

    wikitext_ds = load_wikitext(split="test")

    metrics = evaluate_wikitext(model, tokenizer, wikitext_ds, max_length=2048)

    metrics["config"] = {
        "model": args.model_path,
        "dtype": args.dtype,
        "base_eval": bool(args.base_eval),
        "w_bit": args.w_bit,
        "q_group_size": args.q_group_size,
        "no_zero_point": bool(args.no_zero_point),
        "q_backend": args.q_backend,
    }

    print("\n=== WikiText Metrics ===")
    print(json.dumps(metrics, indent=2))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[WikiText] Metrics written to {args.output_path}")


if __name__ == "__main__":
    main()
