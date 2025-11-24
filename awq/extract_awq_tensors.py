"""Utilities for exporting AWQ scales, weights, and scale-weight products.

This helper script loads a transformer model alongside its AWQ cache
and writes out per-layer tensors so they can be inspected or visualized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# Make the bundled llm-awq sources importable when running this script directly.
ROOT_DIR = Path(__file__).resolve().parent
AWQ_SRC = ROOT_DIR / "llm-awq"
if AWQ_SRC.exists():
    sys.path.insert(0, str(AWQ_SRC))

from awq.quantize.pre_quant import get_named_linears  # noqa: E402


def load_model(model_path: str, torch_dtype: str = "float16", device_map: str = "auto"):
    """Load a causal LM with the given dtype and device map."""
    dtype = getattr(torch, torch_dtype)
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map)


def load_awq_cache(cache_path: str) -> Dict:
    """Load an AWQ cache produced by the llm-awq quantizer."""
    return torch.load(cache_path)


def sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def ensure_output_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_normalized_awq_weights(
    model,
    awq_results: Dict,
    output_dir: str | Path,
    bit_width: int,
) -> Iterable[Tuple[str, Path]]:
    """Extract normalized AWQ-scaled weights for each linear layer.

    Parameters
    ----------
    model:
        Loaded transformer model.
    awq_results:
        Object returned by ``torch.load`` on an AWQ cache file.
    output_dir:
        Folder where ``*.npy`` files will be written.
    bit_width:
        Quantization bit width used to compute the step size.

    Yields
    ------
    tuple
        ``(layer_name, normalized_weight_path)`` for each processed layer.
    """

    linears = get_named_linears(model)
    state_dict = model.state_dict()
    scales = awq_results.get("scale", [])
    out_dir = ensure_output_dir(output_dir)
    denom = (2 ** (bit_width - 1)) - 1

    for block_name, scale_names, scale_tensor in scales:
        if scale_tensor.ndim == 0:
            continue

        scale_vector = scale_tensor.detach().cpu().float().numpy().reshape(-1)

        for scale_name in scale_names:
            linear_layer = linears.get(scale_name)
            if linear_layer is None:
                continue

            weight_param = state_dict.get(f"{scale_name}.weight")
            if weight_param is None:
                continue
            if weight_param.is_meta:
                raise RuntimeError(
                    f"Layer {scale_name} has meta weights; reload the model with real tensors (e.g., --device-map cpu)."
                )
            weight_matrix = weight_param.detach().cpu().float().numpy()

            if weight_matrix.shape[1] != scale_vector.shape[0]:
                # Mismatched shapes are unexpected; skip to avoid silent broadcasting errors.
                continue

            scale_weight_product = weight_matrix * scale_vector.reshape(1, -1)
            max_abs = np.max(np.abs(scale_weight_product), axis=0)
            step_sizes = max_abs / denom
            step_sizes[step_sizes == 0] = 1.0
            normalized_weights = scale_weight_product / step_sizes.reshape(1, -1)

            layer_prefix = out_dir / sanitize_name(scale_name)
            normalized_path = layer_prefix.with_name(layer_prefix.name + "_normalized.npy")
            np.save(normalized_path, normalized_weights)

            yield scale_name, normalized_path


def extract_awq_weights(
    model,
    awq_results: Dict,
    output_dir: str | Path,
) -> Iterable[Tuple[str, Path]]:
    """Dump raw weights and their corresponding AWQ scales for each linear submodule."""

    state_dict = model.state_dict()
    linears = get_named_linears(model)
    scales = awq_results.get("scale", [])
    out_dir = ensure_output_dir(output_dir)

    for _, scale_names, scale_tensor in scales:
        if scale_tensor.ndim == 0:
            continue

        awq_scale = scale_tensor.detach().cpu().float().view(-1)

        for scale_name in scale_names:
            linear_layer = linears.get(scale_name)
            if linear_layer is None:
                continue

            weight_param = state_dict.get(f"{scale_name}.weight")
            if weight_param is None:
                continue
            if weight_param.is_meta:
                raise RuntimeError(
                    f"Layer {scale_name} has meta weights; reload the model with real tensors (e.g., --device-map cpu)."
                )
            weight_tensor = weight_param.detach().cpu().float()
            if weight_tensor.shape[1] != awq_scale.shape[0]:
                continue

            payload = {
                "weights": weight_tensor.clone(),
                "awq_scale": awq_scale.clone(),
            }

            layer_prefix = out_dir / sanitize_name(scale_name)
            weights_path = layer_prefix.with_name(layer_prefix.name + "_awq.pt")
            torch.save(payload, weights_path)

            yield scale_name, weights_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path or HF model id for the base model.")
    parser.add_argument("--awq-cache", required=True, help="Path to the AWQ cache (e.g., awq_cache/llama3-8b-w4-g128.pt).")
    parser.add_argument(
        "--output-dir",
        default="awq_extracted",
        help="Directory where extracted tensors will be written (default: awq_extracted).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Torch dtype string for loading the model (default: float16).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers for model loading (default: auto).",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        default=4,
        help="Bit width of the signed quantizer used for AWQ (default: 4).",
    )
    parser.add_argument(
        "--extract-awq",
        action="store_true",
        help="Only dump raw AWQ weights/scales. If neither extraction flag is provided, both pipelines run.",
    )
    parser.add_argument(
        "--extract-normalized",
        action="store_true",
        help="Only dump normalized AWQ weights. If neither extraction flag is provided, both pipelines run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model, torch_dtype=args.torch_dtype, device_map=args.device_map)
    awq_results = load_awq_cache(args.awq_cache)

    summary_map: Dict[str, Dict[str, str]] = {}

    run_awq_dump = args.extract_awq or (not args.extract_awq and not args.extract_normalized)
    run_normalized_dump = args.extract_normalized or (not args.extract_awq and not args.extract_normalized)

    if run_awq_dump:
        for layer_name, weights_path in extract_awq_weights(model, awq_results, args.output_dir):
            entry = summary_map.setdefault(layer_name, {"layer": layer_name})
            entry["awq_weights"] = str(weights_path)

    if run_normalized_dump:
        for layer_name, normalized_path in extract_normalized_awq_weights(
            model, awq_results, args.output_dir, bit_width=args.bit_width
        ):
            entry = summary_map.setdefault(layer_name, {"layer": layer_name})
            entry["normalized"] = str(normalized_path)

    summary = list(summary_map.values())

    summary_path = ensure_output_dir(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
