#!/usr/bin/env python3
"""
Layer-wise weight distribution visualizer for Meta-Llama-3-8B.

For each transformer block (model.model.layers[i]), this script:
  1) Reservoir-samples up to N weights across all submodules in the layer
  2) Plots a histogram of sampled weights
  3) Plots Q-Q comparisons against Normal, Laplace, and Logistic distributions
  4) Saves one PNG per layer

Usage (CPU/GPU auto offload):
  python llama_weight_distributions.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --sample_per_layer 2000000 \
    --bins 200 \
    --dtype bfloat16 \
    --device_map auto \
    --outdir ./llama3_weight_plots

If you lack GPU memory, prefer --device_map cpu and reduce --sample_per_layer.
"""

import argparse
import math
import os
import random
import sys
from typing import Iterable, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise weight histograms and QQ plots for Llama.")
    p.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B",
                   help="Hugging Face model id or local path.")
    p.add_argument("--revision", type=str, default=None, help="Optional HF revision/commit/tag.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"],
                   help="Torch dtype to load the model weights.")
    p.add_argument("--device_map", type=str, default="auto", choices=["auto", "cpu"],
                   help="accelerate-style device map. 'auto' to shard to GPU/CPU if available, 'cpu' to force CPU.")
    p.add_argument("--sample_per_layer", type=int, default=2_000_000,
                   help="Max number of weights to sample per layer (reservoir sampling).")
    p.add_argument("--bins", type=int, default=200, help="Histogram bins.")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    p.add_argument("--outdir", type=str, default="./llama_weight_plots",
                   help="Output directory for PNGs.")
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
    """
    Yield flattened weight arrays (numpy) for each parameter tensor in the layer.
    Biases are included (they're informative and small).
    Grad is disabled; we only read .data.
    """
    with torch.no_grad():
        for _, p in layer.named_parameters(recurse=True):
            # Skip empty or None
            if p is None or p.numel() == 0:
                continue
            # Some params may be on GPU due to device_map=auto; move chunk-wise to CPU as numpy
            # NOTE: .float() for consistent QQ against continuous distributions
            arr = p.detach().to("cpu").float().view(-1).numpy()
            if arr.size == 0:
                continue
            yield arr


def reservoir_sample_array_stream(stream: Iterable[np.ndarray], k: int, rng: random.Random) -> np.ndarray:
    """
    Reservoir-sample exactly up to k elements from a stream of 1D numpy arrays
    without holding the entire stream in memory.

    Returns a 1D numpy array with size <= k.
    """
    if k <= 0:
        return np.empty((0,), dtype=np.float32)

    # Initialize reservoir
    reservoir = None
    filled = 0
    seen = 0

    for chunk in stream:
        # If still filling the reservoir
        if reservoir is None:
            if chunk.size >= k:
                # Take the first k and then handle the rest of the chunk via standard reservoir logic
                reservoir = np.array(chunk[:k], dtype=np.float32, copy=True)
                filled = k
                seen = k
                remainder = chunk[k:]
                # Now process the remainder with reservoir updates
                for val in remainder:
                    seen += 1
                    j = rng.randint(1, seen)
                    if j <= k:
                        reservoir[j - 1] = val
            else:
                # Take all; might need to expand later
                reservoir = np.array(chunk, dtype=np.float32, copy=True)
                filled = chunk.size
                seen = chunk.size
        else:
            # We already have a reservoir
            for val in chunk:
                seen += 1
                if filled < k:
                    # Still space to fill
                    reservoir = np.append(reservoir, np.float32(val))
                    filled += 1
                else:
                    j = rng.randint(1, seen)
                    if j <= k:
                        reservoir[j - 1] = val

    if reservoir is None:
        return np.empty((0,), dtype=np.float32)
    return reservoir



def make_hist_and_qq(layer_idx: int, sample: np.ndarray, bins: int, outdir: str):
    """
    Top: histogram of weights.
    Bottom: single Q–Q plot overlaying Normal, Laplace, Logistic theoretical quantiles
            against empirical (standardized) sample quantiles. Includes a legend.
    Saves to outdir/layer_{idx:02d}_weights.png
    """
    if sample.size == 0:
        print(f"[Layer {layer_idx}] No weights found; skipping.")
        return

    # --- Prepare empirical standardized quantiles ---
    x = np.sort(sample.astype(np.float64))
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        print(f"[Layer {layer_idx}] Degenerate variance; skipping QQ.", file=sys.stderr)
        return

    z = (x - mu) / sigma  # empirical z-quantiles
    n = z.size
    # Use plotting positions (i - 0.5)/n for stability
    p = (np.arange(1, n + 1) - 0.5) / n

    # --- Theoretical quantiles with unit variance ---
    # Normal: std=1 already
    q_norm = stats.norm.ppf(p, loc=0, scale=1.0)
    # Laplace with unit variance -> scale = 1/sqrt(2)
    q_lap  = stats.laplace.ppf(p, loc=0, scale=1.0 / np.sqrt(2.0))
    # Logistic with unit variance -> scale = sqrt(3)/pi
    q_log  = stats.logistic.ppf(p, loc=0, scale=np.sqrt(3.0) / np.pi)

    # --- Figure layout: 2 rows (hist on top; QQ overlay bottom) ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])

    # Histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(sample, bins=bins, density=True)
    ax_hist.set_title(f"Llama Layer {layer_idx}: Weight Value Histogram (n={sample.size:,})")
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")

    # Q–Q overlay
    ax = fig.add_subplot(gs[1, 0])

    # Reference 45° line: y = x
    # Choose symmetric range that covers all theoretical quantiles
    q_all = np.concatenate([q_norm, q_lap, q_log])
    q_min, q_max = np.percentile(q_all, [0.5, 99.5])
    lim = max(abs(q_min), abs(q_max))
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1, label="y = x (reference)")

    # Plot empirical vs theoretical for each distribution
    # (theoretical on x-axis, empirical standardized on y-axis)
    ax.plot(q_norm, z, linewidth=1.2, alpha=0.9, label="Normal Q–Q")
    ax.plot(q_lap,  z, linewidth=1.2, alpha=0.9, label="Laplace Q–Q")
    ax.plot(q_log,  z, linewidth=1.2, alpha=0.9, label="Logistic Q–Q")

    ax.set_xlim(-lim, lim)
    # Match y-limits to x for fair visual comparison
    z_min, z_max = np.percentile(z, [0.5, 99.5])
    lim_y = max(abs(z_min), abs(z_max), lim)
    ax.set_ylim(-lim_y, lim_y)

    ax.set_title("Q–Q Overlay: Empirical (standardized) vs Theoretical Quantiles")
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical quantile (standardized)")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"layer_{layer_idx:02d}_weights.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[Layer {layer_idx}] Saved {outfile}")




def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = torch_dtype_from_str(args.dtype)

    print(f"Loading config: {args.model_id}")
    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    # Avoids caching KV by default to lower peak memory when we don't need generation
    try:
        config.use_cache = False
    except Exception:
        pass

    print(f"Loading model weights (dtype={dtype}, device_map={args.device_map})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=(None if args.device_map == "cpu" else "auto"),
        trust_remote_code=True,
    )
    model.eval()
    torch.set_grad_enabled(False)

    # Grab transformer layers (Llama-style)
    # Expected: model.model.layers is a ModuleList
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        print("ERROR: Unexpected model structure. Expected model.model.layers to exist.", file=sys.stderr)
        sys.exit(1)

    num_layers = len(model.model.layers)
    print(f"Found {num_layers} transformer layers.")

    for i, layer in enumerate(tqdm(model.model.layers, total=num_layers, desc="Processing layers")):
        # Stream all params in this layer and reservoir-sample up to sample_per_layer values
        stream = iter_param_chunks_1d(layer)
        sample = reservoir_sample_array_stream(stream, args.sample_per_layer, rng)
        make_hist_and_qq(i, sample, args.bins, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
