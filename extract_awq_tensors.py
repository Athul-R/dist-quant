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


def extract_scale_weight_products(
    model,
    awq_results: Dict,
    output_dir: str | Path,
    save_weights: bool = False,
) -> Iterable[Tuple[str, Path, Path, Path]]:
    """Extract scales, weights, and scale-weight products for each linear layer.

    Parameters
    ----------
    model:
        Loaded transformer model.
    awq_results:
        Object returned by ``torch.load`` on an AWQ cache file.
    output_dir:
        Folder where ``*.pth`` files will be written.
    save_weights:
        If ``True``, also writes the raw weight matrix for each layer.

    Yields
    ------
    tuple
        ``(layer_name, scale_path, weight_path, scale_weight_path)`` for each processed layer.
    """

    linears = get_named_linears(model)
    scales = awq_results.get("scale", [])
    out_dir = ensure_output_dir(output_dir)

    for block_name, scale_names, scale_tensor in scales:
        if scale_tensor.ndim == 0:
            continue

        scale_vector = scale_tensor.detach().cpu().float().numpy().reshape(-1)

        for scale_name in scale_names:
            linear_layer = linears.get(scale_name)
            if linear_layer is None:
                continue

            weight_matrix = linear_layer.weight.detach().cpu().float().numpy()

            if weight_matrix.shape[1] != scale_vector.shape[0]:
                # Mismatched shapes are unexpected; skip to avoid silent broadcasting errors.
                continue

            scale_weight_product = weight_matrix * scale_vector.reshape(1, -1)

            layer_prefix = out_dir / sanitize_name(scale_name)
            scale_path = layer_prefix.with_name(layer_prefix.name + "_scale.npy")
            scale_weight_path = layer_prefix.with_name(layer_prefix.name + "_scale_weight.npy")
            weight_path = (
                layer_prefix.with_name(layer_prefix.name + "_weight.npy") if save_weights else None
            )

            np.save(scale_path, scale_vector)
            np.save(scale_weight_path, scale_weight_product)
            if save_weights and weight_path is not None:
                np.save(weight_path, weight_matrix)

            yield scale_name, scale_path, weight_path, scale_weight_path


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
        "--save-weights",
        action="store_true",
        help="Also store the raw weight matrices (can be large).",
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
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model, torch_dtype=args.torch_dtype, device_map=args.device_map)
    awq_results = load_awq_cache(args.awq_cache)

    summary = []
    for layer_name, scale_path, weight_path, scale_weight_path in extract_scale_weight_products(
        model, awq_results, args.output_dir, save_weights=args.save_weights
    ):
        summary.append(
            {
                "layer": layer_name,
                "scale": str(scale_path),
                "weight": str(weight_path) if weight_path else None,
                "scale_weight": str(scale_weight_path),
            }
        )

    summary_path = ensure_output_dir(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
