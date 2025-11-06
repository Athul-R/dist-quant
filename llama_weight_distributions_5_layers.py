#!/usr/bin/env python3
"""
Layer-wise weight distribution visualizer for Meta-Llama-3-8B (first 5 layers only).

This script:
  1) Loads the Meta-Llama-3-8B model
  2) Takes the first N transformer layers (default = 5)
  3) Samples weights per layer
  4) Plots histograms and Q-Q plots vs Normal, Laplace, and Logistic distributions
  5) Saves one figure per layer

Example:
  python llama_weight_distributions_5_layers.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --num_layers 5 \
    --sample_per_layer 2000000 \
    --bins 200 \
    --dtype bfloat16 \
    --device_map auto \
    --outdir ./llama3_weight_plots

Dependencies:
  pip install torch transformers scipy matplotlib tqdm
"""

import argparse
import os
import random
import sys
from typing import Iterable

import torch
from transformers import AutoConfig, AutoModelForCausalLM
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Llama-3 weight histograms and QQ plots (first 5 layers).")
    p.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B",
                   help="Hugging Face model ID or local path.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                   help="Torch dtype.")
    p.add_argument("--device_map", type=str, default="auto", choices=["auto", "cpu"],
                   help="Use auto offloading or CPU only.")
    p.add_argument("--num_layers", type=int, default=5,
                   help="Number of transformer layers to analyze (from the start).")
    p.add_argument("--sample_per_layer", type=int, default=2_000_000,
                   help="Number of weights to sample per layer.")
    p.add_argument("--bins", type=int, default=200, help="Histogram bins.")
    p.add_argument("--outdir", type=str, default="./llama_weight_plots_5_layers",
                   help="Output directory for plots.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def torch_dtype_from_str(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def iter_param_chunks_1d(layer: torch.nn.Module) -> Iterable[np.ndarray]:
    """Yield flattened 1D weight arrays from each parameter in the layer."""
    with torch.no_grad():
        for _, p in layer.named_parameters(recurse=True):
            if p is None or p.numel() == 0:
                continue
            arr = p.detach().to("cpu").float().view(-1).numpy()
            if arr.size > 0:
                yield arr


def reservoir_sample(stream: Iterable[np.ndarray], k: int, rng: random.Random) -> np.ndarray:
    """Reservoir-sample up to k elements from a stream of numpy arrays."""
    reservoir = np.empty(0, dtype=np.float32)
    seen = 0

    for chunk in stream:
        for val in chunk:
            seen += 1
            if len(reservoir) < k:
                reservoir = np.append(reservoir, val)
            else:
                j = rng.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = val
    return reservoir


def make_hist_and_qq(layer_idx: int, sample: np.ndarray, bins: int, outdir: str):
    """Plot histogram + QQ plots for one layer."""
    if sample.size == 0:
        print(f"[Layer {layer_idx}] Empty weights, skipping.")
        return

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.hist(sample, bins=bins, density=True)
    ax_hist.set_title(f"Llama Layer {layer_idx}: Weight Distribution")
    ax_hist.set_xlabel("Weight Value")
    ax_hist.set_ylabel("Density")

    dists = [("Normal", "norm"), ("Laplace", "laplace"), ("Logistic", "logistic")]
    for j, (name, dist_name) in enumerate(dists):
        ax = fig.add_subplot(gs[1, j % 2])
        stats.probplot(sample, dist=getattr(stats, dist_name), plot=ax)
        ax.set_title(f"Q-Q vs {name}")

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"layer_{layer_idx:02d}.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[Layer {layer_idx}] Saved {outfile}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch_dtype_from_str(args.dtype)

    print(f"Loading {args.model_id}...")
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    try:
        config.use_cache = False
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=(None if args.device_map == "cpu" else "auto"),
        trust_remote_code=True,
    )
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        print("Error: model.model.layers not found.")
        sys.exit(1)

    total_layers = len(model.model.layers)
    n = min(args.num_layers, total_layers)
    print(f"Processing first {n} of {total_layers} layers...")

    for i, layer in enumerate(tqdm(model.model.layers[:n], total=n)):
        stream = iter_param_chunks_1d(layer)
        sample = reservoir_sample(stream, args.sample_per_layer, rng)
        make_hist_and_qq(i, sample, args.bins, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
