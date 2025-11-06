#!/usr/bin/env python3
"""
Layer-wise weight distribution visualizer for Meta-Llama-3-8B (safetensors via PyTorch; no model load).

- Reads *.safetensors shards directly (avoids accelerate/meta tensor issues).
- Uniformly samples up to N weights per layer across keys matching "model.layers.{i}.*".
- Plots histogram + single Q–Q overlay vs Normal/Laplace/Logistic (unit variance).
- Computes Quantile RMSE **and** MAE per layer and prints them.
- Adds MAE to the legend and shows the **parameter count per layer** in each plot title.
- Prints mean ± std over layers for RMSE and MAE (for layers that produced samples) and writes a summary file.
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy import stats
from tqdm import tqdm

import torch
from safetensors import safe_open
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Layer-wise weight histograms and QQ overlays (no model load).")
    p.add_argument("--model_id", type=str, required=True,
                   help="Local path to folder with *.safetensors (or HF repo id if not using --local_only).")
    p.add_argument("--revision", type=str, default=None, help="Optional HF revision/commit/tag (if not --local_only).")
    p.add_argument("--local_only", action="store_true",
                   help="Do not download; treat --model_id as a local directory with safetensors.")
    p.add_argument("--sample_per_layer", type=int, default=2_000_000,
                   help="Max number of weights to sample per layer (uniform without replacement).")
    p.add_argument("--bins", type=int, default=200, help="Histogram bins.")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    p.add_argument("--outdir", type=str, default="./llama3_weight_plots",
                   help="Output directory for PNGs.")
    return p.parse_args()


# ----------------------------
# Utils
# ----------------------------
def find_snapshot_path(model_id: str, revision: str | None, local_only: bool) -> str:
    if local_only:
        if not os.path.isdir(model_id):
            print(f"ERROR: --local_only specified, but path not found: {model_id}", file=sys.stderr)
            sys.exit(1)
        return os.path.abspath(model_id)
    if snapshot_download is None:
        print("ERROR: huggingface_hub not installed; use --local_only or install it.", file=sys.stderr)
        sys.exit(1)
    return snapshot_download(repo_id=model_id, revision=revision, allow_patterns=["*.safetensors"])


def list_safetensors(root: str) -> List[str]:
    out = [os.path.join(root, fn) for fn in os.listdir(root) if fn.endswith(".safetensors")]
    if not out:
        print(f"ERROR: No *.safetensors found under {root}", file=sys.stderr)
        sys.exit(1)
    return sorted(out)


# ----------------------------
# Sampling from shards (PyTorch backend)
# ----------------------------
def sample_from_tensor_pt(shard_path: str, key: str, num: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly sample 'num' elements from the tensor at (shard_path, key) using PyTorch backend.
    Returns float32 numpy array on CPU. Only the sampled slice is materialized.
    """
    with safe_open(shard_path, framework="pt") as f:
        t: torch.Tensor = f.get_tensor(key)  # CPU tensor (bf16/fp16/fp32)
        flat = t.view(-1)
        n = flat.numel()
        if n == 0 or num <= 0:
            return np.empty((0,), dtype=np.float32)
        take = min(n, num)
        if take == n:
            return flat.to(dtype=torch.float32).cpu().numpy()
        idx_np = rng.choice(n, size=take, replace=False)
        idx = torch.from_numpy(idx_np).to(dtype=torch.long)  # CPU
        gathered = torch.index_select(flat, 0, idx).to(dtype=torch.float32)
        return gathered.cpu().numpy()


def sample_layer_uniform_from_shards(shard_paths: List[str], layer_idx: int, k: int,
                                     rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Uniformly sample up to k elements from ALL tensors whose key starts with model.layers.{layer_idx}.
    Proportional allocation with stochastic rounding (effectively uniform overall).
    Returns:
        sample (np.ndarray float32), total_params_in_layer (int)
    """
    tensors: List[Tuple[str, str, int]] = []  # (shard_path, key, numel)
    total = 0

    # Discover tensors & sizes
    for shard in shard_paths:
        with safe_open(shard, framework="pt") as f:
            for kname in f.keys():
                if not kname.startswith(f"model.layers.{layer_idx}."):
                    continue
                t = f.get_tensor(kname)  # view
                n = t.numel()
                if n > 0:
                    tensors.append((shard, kname, n))
                    total += n

    if total == 0 or k <= 0:
        return np.empty((0,), dtype=np.float32), int(total)

    k_eff = min(k, total)

    # Proportional allocation with stochastic rounding
    shares = [(shard, key, n, (n / total) * k_eff) for (shard, key, n) in tensors]
    floor_parts = [(shard, key, n, int(math.floor(exp))) for (shard, key, n, exp) in shares]
    taken = sum(fp[3] for fp in floor_parts)
    remainder = k_eff - taken

    bump = set()
    if remainder > 0:
        fracs = sorted(
            [(exp - math.floor(exp), shard, key, n) for (shard, key, n, exp) in shares],
            key=lambda x: x[0],
            reverse=True
        )
        for i in range(remainder):
            _, shard, key, _ = fracs[i]
            bump.add((shard, key))

    samples = []
    for (shard, key, n, base_take) in floor_parts:
        take = base_take + (1 if (shard, key) in bump else 0)
        if take <= 0:
            continue
        samples.append(sample_from_tensor_pt(shard, key, take, rng))

    if not samples:
        return np.empty((0,), dtype=np.float32), int(total)
    return np.concatenate(samples).astype(np.float32, copy=False), int(total)


# ----------------------------
# Plot + RMSE + MAE
# ----------------------------
def make_hist_and_qq(layer_idx: int, sample: np.ndarray, bins: int, outdir: str, n_params: int) -> Tuple[float, float, float, float, float, float]:
    """
    Returns:
      (rmse_norm, rmse_lap, rmse_log, mae_norm, mae_lap, mae_log).
    If no valid sample, returns (nan, nan, nan, nan, nan, nan).
    """
    if sample.size == 0:
        print(f"[Layer {layer_idx}] No weights found; skipping.")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    x = np.sort(sample.astype(np.float64))
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        print(f"[Layer {layer_idx}] Degenerate variance; skipping.", file=sys.stderr)
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    z = (x - mu) / sigma
    n = z.size
    p = (np.arange(1, n + 1) - 0.5) / n

    # Unit-variance theoretical quantiles
    q_norm = stats.norm.ppf(p, loc=0, scale=1.0)
    q_lap  = stats.laplace.ppf(p, loc=0, scale=1.0 / np.sqrt(2.0))
    q_log  = stats.logistic.ppf(p, loc=0, scale=np.sqrt(3.0) / np.pi)

    # Errors in quantile space
    eN = z - q_norm
    eL = z - q_lap
    eG = z - q_log

    # RMSE
    rmse_norm = float(np.sqrt(np.mean(eN ** 2)))
    rmse_lap  = float(np.sqrt(np.mean(eL ** 2)))
    rmse_log  = float(np.sqrt(np.mean(eG ** 2)))
    # MAE
    mae_norm = float(np.mean(np.abs(eN)))
    mae_lap  = float(np.mean(np.abs(eL)))
    mae_log  = float(np.mean(np.abs(eG)))

    print(f"[Layer {layer_idx:02d}] "
          f"RMSE  N={rmse_norm:.4f}  L={rmse_lap:.4f}  G={rmse_log:.4f}  |  "
          f"MAE  N={mae_norm:.4f}  L={mae_lap:.4f}  G={mae_log:.4f}")

    # Determine best (smallest RMSE) for title
    rmse_vals = {"Normal": rmse_norm, "Laplace": rmse_lap, "Logistic": rmse_log}
    best_name = min(rmse_vals, key=rmse_vals.get)
    best_rmse = rmse_vals[best_name]

    # Figure
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])

    # Histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(sample, bins=bins, density=True)
    ax_hist.set_title(
        f"Llama Layer {layer_idx} • Params={n_params:,} • Sample n={sample.size:,} • "
        f"Best (RMSE): {best_name} ({best_rmse:.4f})"
    )
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")

    # Q–Q overlay
    ax = fig.add_subplot(gs[1, 0])
    q_all = np.concatenate([q_norm, q_lap, q_log])
    q_min, q_max = np.percentile(q_all, [0.5, 99.5])
    lim = float(max(abs(q_min), abs(q_max)))
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1, label="y = x (reference)")

    # Legend now contains RMSE **and** MAE
    ax.plot(q_norm, z, linewidth=1.2, alpha=0.9,
            label=f"Normal Q–Q (RMSE={rmse_norm:.4f}, MAE={mae_norm:.4f})")
    ax.plot(q_lap,  z, linewidth=1.2, alpha=0.9,
            label=f"Laplace Q–Q (RMSE={rmse_lap:.4f}, MAE={mae_lap:.4f})")
    ax.plot(q_log,  z, linewidth=1.2, alpha=0.9,
            label=f"Logistic Q–Q (RMSE={rmse_log:.4f}, MAE={mae_log:.4f})")

    ax.set_xlim(-lim, lim)
    z_min, z_max = np.percentile(z, [0.5, 99.5])
    lim_y = float(max(abs(z_min), abs(z_max), lim))
    ax.set_ylim(-lim_y, lim_y)

    ax.set_title("Q–Q Overlay: Empirical (standardized) vs Theoretical Quantiles")
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical quantile (standardized)")
    ax.legend(loc="best", frameon=True)

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"layer_{layer_idx:02d}_weights.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[Layer {layer_idx}] Saved {outfile}")

    return (rmse_norm, rmse_lap, rmse_log, mae_norm, mae_lap, mae_log)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    snapshot_path = find_snapshot_path(args.model_id, args.revision, local_only=args.local_only)
    shard_paths = list_safetensors(snapshot_path)
    print(f"Using {len(shard_paths)} safetensors shard(s) from: {snapshot_path}")

    # Infer #layers
    max_layer = -1
    for sp in shard_paths:
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.layers."):
                    parts = k.split(".")
                    if len(parts) > 2 and parts[2].isdigit():
                        idx = int(parts[2])
                        if idx > max_layer:
                            max_layer = idx
    num_layers = (max_layer + 1) if max_layer >= 0 else 32
    print(f"Found {num_layers} transformer layers (by scanning shard keys).")

    rmse_norm_all, rmse_lap_all, rmse_log_all = [], [], []
    mae_norm_all, mae_lap_all, mae_log_all = [], [], []
    counted_layers = 0

    for i in tqdm(range(num_layers), desc="Processing layers"):
        sample, n_params = sample_layer_uniform_from_shards(shard_paths, i, args.sample_per_layer, rng)
        if sample.size == 0 or n_params == 0:
            print(f"[Layer {i}] No weights found in shards; skipping.")
            continue
        rn, rl, rlg, an, al, alg = make_hist_and_qq(i, sample, args.bins, args.outdir, n_params)
        if all(np.isfinite(v) for v in (rn, rl, rlg, an, al, alg)):
            rmse_norm_all.append(rn); rmse_lap_all.append(rl); rmse_log_all.append(rlg)
            mae_norm_all.append(an);  mae_lap_all.append(al);  mae_log_all.append(alg)
            counted_layers += 1

    # ---- Summary stats over layers ----
    def mean_std(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return (float("nan"), float("nan"))
        if arr.size == 1:
            return (float(arr[0]), 0.0)
        return (float(np.mean(arr)), float(np.std(arr, ddof=1)))

    if counted_layers == 0:
        print("No layers produced valid samples; no summary metrics.")
    else:
        mN, sN = mean_std(rmse_norm_all)
        mL, sL = mean_std(rmse_lap_all)
        mG, sG = mean_std(rmse_log_all)

        aN, aNs = mean_std(mae_norm_all)
        aL, aLs = mean_std(mae_lap_all)
        aG, aGs = mean_std(mae_log_all)

        print("\n==== Quantile RMSE summary over layers ====")
        print(f"Layers counted: {counted_layers}/{num_layers}")
        print(f"Normal  : mean={mN:.6f}, std={sN:.6f}")
        print(f"Laplace : mean={mL:.6f}, std={sL:.6f}")
        print(f"Logistic: mean={mG:.6f}, std={sG:.6f}")

        print("\n==== Quantile MAE summary over layers ====")
        print(f"Normal  : mean={aN:.6f}, std={aNs:.6f}")
        print(f"Laplace : mean={aL:.6f}, std={aLs:.6f}")
        print(f"Logistic: mean={aG:.6f}, std={aGs:.6f}")

        # Write a summary file
        os.makedirs(args.outdir, exist_ok=True)
        summary_path = os.path.join(args.outdir, "rmse_mae_summary.txt")
        with open(summary_path, "w") as fh:
            fh.write(f"Layers counted: {counted_layers}/{num_layers}\n\n")
            fh.write("RMSE (quantile-space):\n")
            fh.write(f"  Normal  : mean={mN:.6f}, std={sN:.6f}\n")
            fh.write(f"  Laplace : mean={mL:.6f}, std={sL:.6f}\n")
            fh.write(f"  Logistic: mean={mG:.6f}, std={sG:.6f}\n\n")
            fh.write("MAE (quantile-space):\n")
            fh.write(f"  Normal  : mean={aN:.6f}, std={aNs:.6f}\n")
            fh.write(f"  Laplace : mean={aL:.6f}, std={aLs:.6f}\n")
            fh.write(f"  Logistic: mean={aG:.6f}, std={aGs:.6f}\n")
        print(f"\nSummary written to {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
