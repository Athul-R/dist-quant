#!/usr/bin/env python3
"""
Layer-wise & submodule-wise weight distribution visualizer for AWQ-extracted tensors.

Supports LLaMA and Qwen families.

Features (enable via flags):
- Per-layer uniform sampling from AWQ ``*.npy`` dumps (no full model load).
- Per-layer histogram + Q–Q overlays vs Normal/Laplace/Logistic (unit variance).
- RMSE & MAE in quantile-space per layer; aggregate mean±std.
- Metrics-vs-layer plot, per-layer CSV, submodule plots (q/k/v/o and gate/up/down),
  torchinfo summary (best-effort, meta tensors), reproducibility banner.
- NEW: Submodule-level metrics CSV across all layers or selected layer indices.

"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from tqdm import tqdm

# Optional deps for torchinfo summary (best-effort)
try:
    from transformers import AutoConfig, AutoModelForCausalLM
except Exception:
    AutoConfig = None
    AutoModelForCausalLM = None

try:
    from accelerate import init_empty_weights
except Exception:
    init_empty_weights = None

try:
    from torchinfo import summary as torchinfo_summary
except Exception:
    torchinfo_summary = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UTC = timezone.utc


# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-wise and submodule-wise weight histograms and QQ overlays (no model load)."
    )

    # Architecture selection (affects default outdir; regex are generic enough for both)
    p.add_argument(
        "--arch", type=str, choices=["llama", "qwen"], default="llama",
        help="Model family preset; used mainly for default output folder."
    )

    p.add_argument("--model_id", type=str, required=True,
                   help="Local path with *.safetensors or HF repo id.")
    p.add_argument("--revision", type=str, default=None,
                   help="Optional HF revision/commit/tag (used for torchinfo summary).")

    p.add_argument("--sample_per_layer", type=int, default=2_000_000,
                   help="Max weights sampled per layer (uniform without replacement).")
    p.add_argument("--submodule_sample", type=int, default=1_000_000,
                   help="Max weights sampled per submodule (uniform without replacement).")
    p.add_argument("--bins", type=int, default=200, help="Histogram bins.")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")

    # If not provided, will be chosen based on --arch
    p.add_argument("--outdir", type=str, default="",
                   help="Output directory. If empty, set to arch-specific default.")

    # Fine-grained toggles
    p.add_argument("--do_layer_overall_plots", action="store_true",
                   help="Generate per-layer overall hist+QQ plots.")
    p.add_argument("--do_layer_metrics_csv", action="store_true",
                   help="Write per-layer metrics CSV.")
    p.add_argument("--do_metrics_vs_layer_plot", action="store_true",
                   help="Generate RMSE/MAE vs layer index plot.")
    p.add_argument("--do_submodule_plots", action="store_true",
                   help="Generate submodule plots for selected layers.")
    p.add_argument("--do_torchinfo", action="store_true",
                   help="Attempt to write a torchinfo-style summary (best-effort).")
    p.add_argument("--do_summary", action="store_true",
                   help="Write reproducibility + aggregate metrics summary file.")

    # NEW: Submodule metrics CSV
    p.add_argument("--do_submodule_metrics_csv", action="store_true",
                   help="Write submodule-level metrics CSV (q,k,v,o,gate_proj,up_proj,down_proj).")
    p.add_argument("--submodule_metrics_all", action="store_true",
                   help="If set with --do_submodule_metrics_csv, compute for ALL layers; otherwise use --sub_layers.")

    # Submodule layers selection (used for plots and for metrics CSV if --submodule_metrics_all is not set)
    p.add_argument("--sub_layers", type=str, default="0,8,16,24,31",
                   help="Comma-separated layer indices for submodule plots / metrics (when not using --submodule_metrics_all).")

    # Torchinfo depth
    p.add_argument("--torchinfo_depth", type=int, default=4,
                   help="Depth for torchinfo summary if enabled.")
    p.add_argument(
        "--awq-dir",
        type=str,
        default="awq_extracted",
        help="Directory containing AWQ-extracted *.npy tensors (default: awq_extracted).",
    )
    return p.parse_args()


# ----------------------------
# Patterns (tolerant for LLaMA/Qwen/GPT-ish)
# ----------------------------
ATTN_PATTERNS = {
    "q": re.compile(r"(?:^|\.)(?:self_attn|attention|attn|mha|transformer\.h\.\d+\.attn)\.(?:q_proj|wq|q)(?:\.|$)"),
    "k": re.compile(r"(?:^|\.)(?:self_attn|attention|attn|mha|transformer\.h\.\d+\.attn)\.(?:k_proj|wk|k)(?:\.|$)"),
    "v": re.compile(r"(?:^|\.)(?:self_attn|attention|attn|mha|transformer\.h\.\d+\.attn)\.(?:v_proj|wv|v)(?:\.|$)"),
    "o": re.compile(r"(?:^|\.)(?:self_attn|attention|attn|mha|transformer\.h\.\d+\.attn)\.(?:o_proj|wo|out_proj|c_proj)(?:\.|$)"),
}
MLP_PATTERNS = {
    "gate_proj": re.compile(r"(?:^|\.)(?:mlp|feed_forward|ffn)\.(?:gate_proj|proj_gate)(?:\.|$)"),
    "up_proj":   re.compile(r"(?:^|\.)(?:mlp|feed_forward|ffn)\.(?:up_proj|proj_up)(?:\.|$)"),
    "down_proj": re.compile(r"(?:^|\.)(?:mlp|feed_forward|ffn)\.(?:down_proj|proj_down|fc_out)(?:\.|$)"),
}
ALL_SUBMODULES = ["q", "k", "v", "o", "gate_proj", "up_proj", "down_proj"]
LAYER_INDEX_PATTERN = re.compile(r"model\.layers\.(\d+)\.")


@dataclass
class AwqTensorInfo:
    layer_idx: int
    full_name: str
    path: Path
    numel: int


# ----------------------------
# Utils for AWQ-extracted tensors
# ----------------------------
def _resolve_entry_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # Allow entries like "awq_extracted/..." by stripping the base dir name.
    parts = list(path.parts)
    if parts and Path(parts[0]).name == base_dir.name:
        parts = parts[1:]
    resolved = base_dir.joinpath(*parts) if parts else base_dir
    return resolved.resolve()


def _parse_layer_index(layer_name: str) -> Optional[int]:
    match = LAYER_INDEX_PATTERN.search(layer_name)
    if match is None:
        return None
    return int(match.group(1))


def load_awq_tensor_infos(awq_dir: str) -> List[AwqTensorInfo]:
    base_dir = Path(awq_dir).expanduser().resolve()
    summary_path = base_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: Expected summary.json under {base_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(summary_path.read_text())
    except Exception as exc:
        print(f"ERROR: Failed to parse {summary_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    infos: List[AwqTensorInfo] = []
    for record in data:
        layer_name = record.get("layer")
        if not layer_name:
            continue
        path_str = record.get("normalized") or record.get("scale_weight") or record.get("weight")
        if not path_str:
            continue
        tensor_path = _resolve_entry_path(base_dir, path_str)
        if not tensor_path.exists():
            print(f"WARNING: Skipping missing tensor path {tensor_path}", file=sys.stderr)
            continue
        layer_idx = _parse_layer_index(layer_name)
        if layer_idx is None:
            print(f"WARNING: Could not infer layer index from name '{layer_name}'; skipping.", file=sys.stderr)
            continue
        arr = np.load(tensor_path, mmap_mode="r")
        numel = int(arr.size)
        infos.append(AwqTensorInfo(layer_idx=layer_idx, full_name=layer_name, path=tensor_path, numel=numel))
        del arr

    if not infos:
        print(f"ERROR: No usable tensors found in {base_dir}", file=sys.stderr)
        sys.exit(1)
    return infos


def sample_from_npy(path: Path, num: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    flat = arr.reshape(-1)
    n = flat.size
    if n == 0 or num <= 0:
        return np.empty((0,), dtype=np.float32)
    take = min(n, num)
    if take == n:
        return np.asarray(flat, dtype=np.float32)
    idx = rng.choice(n, size=take, replace=False)
    sampled = flat[idx]
    return np.asarray(sampled, dtype=np.float32)


def proportional_sample_from_infos(infos: Sequence[AwqTensorInfo], k: int, rng: np.random.Generator) -> np.ndarray:
    total = sum(info.numel for info in infos)
    if total == 0 or k <= 0:
        return np.empty((0,), dtype=np.float32)
    k_eff = min(k, total)
    shares = [(info, (info.numel / total) * k_eff) for info in infos]
    floor_parts = [(info, int(math.floor(expect))) for info, expect in shares]
    taken = sum(take for (_, take) in floor_parts)
    remainder = k_eff - taken

    bump: set[int] = set()
    if remainder > 0:
        fracs = sorted(
            [(expect - math.floor(expect), idx) for idx, (_, expect) in enumerate(shares)],
            key=lambda x: x[0],
            reverse=True,
        )
        for frac, idx in fracs[:remainder]:
            bump.add(idx)

    chunks = []
    for idx, (info, base_take) in enumerate(floor_parts):
        take = base_take + (1 if idx in bump else 0)
        if take <= 0:
            continue
        chunks.append(sample_from_npy(info.path, take, rng))
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def gather_layer_infos(infos: Sequence[AwqTensorInfo], layer_idx: int) -> Tuple[List[AwqTensorInfo], int]:
    matched = [info for info in infos if info.layer_idx == layer_idx]
    total = sum(info.numel for info in matched)
    return matched, total


def sample_layer_uniform_from_awq(
    infos: Sequence[AwqTensorInfo],
    layer_idx: int,
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    layer_infos, total = gather_layer_infos(infos, layer_idx)
    if total == 0:
        return np.empty((0,), dtype=np.float32), 0
    sample = proportional_sample_from_infos(layer_infos, k, rng)
    return sample, total


# ----------------------------
# Metrics helpers
# ----------------------------
def compute_quantile_metrics(z: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    n = z.size
    p = (np.arange(1, n + 1) - 0.5) / n
    q_norm = stats.norm.ppf(p, loc=0, scale=1.0)
    q_lap  = stats.laplace.ppf(p, loc=0, scale=1.0 / np.sqrt(2.0))
    q_log  = stats.logistic.ppf(p, loc=0, scale=np.sqrt(3.0) / np.pi)
    eN, eL, eG = z - q_norm, z - q_lap, z - q_log
    rmse_norm = float(np.sqrt(np.mean(eN ** 2)))
    rmse_lap  = float(np.sqrt(np.mean(eL ** 2)))
    rmse_log  = float(np.sqrt(np.mean(eG ** 2)))
    mae_norm = float(np.mean(np.abs(eN)))
    mae_lap  = float(np.mean(np.abs(eL)))
    mae_log  = float(np.mean(np.abs(eG)))
    return rmse_norm, rmse_lap, rmse_log, mae_norm, mae_lap, mae_log


def plot_hist_and_qq_on_axes(sample: np.ndarray, bins: int, title: str, ax_hist, ax_qq):
    x = np.sort(sample.astype(np.float64))
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        ax_hist.text(0.5, 0.5, "Degenerate variance", ha="center", va="center", transform=ax_hist.transAxes)
        ax_qq.text(0.5, 0.5, "Degenerate variance", ha="center", va="center", transform=ax_qq.transAxes)
        return None

    z = (x - mu) / sigma
    rn, rl, rlg, an, al, alg = compute_quantile_metrics(z)

    # Histogram
    ax_hist.hist(sample, bins=bins, density=True)
    ax_hist.set_title(title)
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")

    # QQ overlay
    n = z.size
    p = (np.arange(1, n + 1) - 0.5) / n
    q_norm = stats.norm.ppf(p, loc=0, scale=1.0)
    q_lap  = stats.laplace.ppf(p, loc=0, scale=1.0 / np.sqrt(2.0))
    q_log  = stats.logistic.ppf(p, loc=0, scale=np.sqrt(3.0) / np.pi)
    q_all = np.concatenate([q_norm, q_lap, q_log])
    q_min, q_max = np.percentile(q_all, [0.5, 99.5])
    lim = float(max(abs(q_min), abs(q_max)))
    ax_qq.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1, label="y=x")

    ax_qq.plot(q_norm, z, linewidth=1.2, alpha=0.9,
               label=f"Normal (RMSE={rn:.4f}, MAE={an:.4f})")
    ax_qq.plot(q_lap,  z, linewidth=1.2, alpha=0.9,
               label=f"Laplace (RMSE={rl:.4f}, MAE={al:.4f})")
    ax_qq.plot(q_log,  z, linewidth=1.2, alpha=0.9,
               label=f"Logistic (RMSE={rlg:.4f}, MAE={alg:.4f})")
    ax_qq.set_xlim(-lim, lim)
    z_min, z_max = np.percentile(z, [0.5, 99.5])
    lim_y = float(max(abs(z_min), abs(z_max), lim))
    ax_qq.set_ylim(-lim_y, lim_y)
    ax_qq.set_title("Q–Q Overlay")
    ax_qq.set_xlabel("Theoretical quantile")
    ax_qq.set_ylabel("Empirical (standardized)")
    ax_qq.legend(loc="best", frameon=True)

    return dict(rmse_normal=rn, rmse_laplace=rl, rmse_logistic=rlg,
                mae_normal=an,  mae_laplace=al,  mae_logistic=alg)


def make_hist_and_qq(layer_idx: int, sample: np.ndarray, bins: int, outdir: str, n_params: int) -> Tuple[float, float, float, float, float, float]:
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
    rn, rl, rlg, an, al, alg = compute_quantile_metrics(z)

    print(f"[Layer {layer_idx:02d}] RMSE N={rn:.4f} L={rl:.4f} G={rlg:.4f} | MAE N={an:.4f} L={al:.4f} G={alg:.4f}")

    rmse_vals = {"Normal": rn, "Laplace": rl, "Logistic": rlg}
    best_name = min(rmse_vals, key=rmse_vals.get)
    best_rmse = rmse_vals[best_name]

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])

    # Hist
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(sample, bins=bins, density=True)
    ax_hist.set_title(
        f"Layer {layer_idx} • Params={n_params:,} • Sample n={sample.size:,} • Best (RMSE): {best_name} ({best_rmse:.4f})"
    )
    ax_hist.set_xlabel("Weight value")
    ax_hist.set_ylabel("Density")

    # QQ
    ax = fig.add_subplot(gs[1, 0])
    n = z.size
    p = (np.arange(1, n + 1) - 0.5) / n
    q_norm = stats.norm.ppf(p, loc=0, scale=1.0)
    q_lap  = stats.laplace.ppf(p, loc=0, scale=1.0 / np.sqrt(2.0))
    q_log  = stats.logistic.ppf(p, loc=0, scale=np.sqrt(3.0) / np.pi)
    q_all = np.concatenate([q_norm, q_lap, q_log])
    q_min, q_max = np.percentile(q_all, [0.5, 99.5])
    lim = float(max(abs(q_min), abs(q_max)))
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1, label="y = x")

    ax.plot(q_norm, z, linewidth=1.2, alpha=0.9,
            label=f"Normal (RMSE={rn:.4f}, MAE={an:.4f})")
    ax.plot(q_lap,  z, linewidth=1.2, alpha=0.9,
            label=f"Laplace (RMSE={rl:.4f}, MAE={al:.4f})")
    ax.plot(q_log,  z, linewidth=1.2, alpha=0.9,
            label=f"Logistic (RMSE={rlg:.4f}, MAE={alg:.4f})")

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

    return (rn, rl, rlg, an, al, alg)


# ----------------------------
# Submodule sampling & plotting & metrics
# ----------------------------
def collect_submodule_infos(
    infos: Sequence[AwqTensorInfo],
    layer_idx: int,
    sub_regex: re.Pattern,
) -> List[AwqTensorInfo]:
    prefix = f"model.layers.{layer_idx}."
    out: List[AwqTensorInfo] = []
    for info in infos:
        if info.layer_idx != layer_idx:
            continue
        tail = info.full_name[len(prefix):] if info.full_name.startswith(prefix) else info.full_name
        if sub_regex.search(tail):
            out.append(info)
    return out


def sample_submodule_from_awq(
    infos: Sequence[AwqTensorInfo],
    layer_idx: int,
    sub_regex: re.Pattern,
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    subset = collect_submodule_infos(infos, layer_idx, sub_regex)
    total = sum(info.numel for info in subset)
    if total == 0:
        return np.empty((0,), dtype=np.float32), 0
    sample = proportional_sample_from_infos(subset, k, rng)
    return sample, total


def plot_submodules_for_layer(
    infos: Sequence[AwqTensorInfo],
    layer_idx: int,
    bins: int,
    sub_k: int,
    rng: np.random.Generator,
    outdir: str,
):
    """
    Two figures:
      - Attention (q,k,v,o): 4 rows x 2 cols (hist, QQ) → submodules/layer_XX_attn_submodules.png
      - MLP (gate, up, down): 3 rows x 2 cols (hist, QQ) → submodules/layer_XX_mlp_submodules.png
    """
    subdir = os.path.join(outdir, "submodules")
    os.makedirs(subdir, exist_ok=True)

    # --- Attention ---
    fig_a = plt.figure(figsize=(16, 18), constrained_layout=True)
    gs_a = fig_a.add_gridspec(4, 2)
    for r, name in enumerate(["q", "k", "v", "o"]):
        ax_h = fig_a.add_subplot(gs_a[r, 0])
        ax_q = fig_a.add_subplot(gs_a[r, 1])
        sample, total = sample_submodule_from_awq(infos, layer_idx, ATTN_PATTERNS[name], sub_k, rng)
        if sample.size == 0:
            ax_h.text(0.5, 0.5, f"No weights for {name}", ha="center", va="center", transform=ax_h.transAxes)
            ax_q.text(0.5, 0.5, f"No weights for {name}", ha="center", va="center", transform=ax_q.transAxes)
            continue
        title = f"Layer {layer_idx} • {name} • Params={total:,} • Sample n={sample.size:,}"
        _ = plot_hist_and_qq_on_axes(sample, bins, title, ax_h, ax_q)

    fig_a.suptitle(f"Attention submodules — Layer {layer_idx}", fontsize=14)
    out_a = os.path.join(subdir, f"layer_{layer_idx:02d}_attn_submodules.png")
    fig_a.savefig(out_a, dpi=150)
    plt.close(fig_a)
    print(f"[Layer {layer_idx}] Saved {out_a}")

    # --- MLP ---
    fig_m = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs_m = fig_m.add_gridspec(3, 2)
    for r, name in enumerate(["gate_proj", "up_proj", "down_proj"]):
        ax_h = fig_m.add_subplot(gs_m[r, 0])
        ax_q = fig_m.add_subplot(gs_m[r, 1])
        sample, total = sample_submodule_from_awq(infos, layer_idx, MLP_PATTERNS[name], sub_k, rng)
        if sample.size == 0:
            ax_h.text(0.5, 0.5, f"No weights for {name}", ha="center", va="center", transform=ax_h.transAxes)
            ax_q.text(0.5, 0.5, f"No weights for {name}", ha="center", va="center", transform=ax_q.transAxes)
            continue
        title = f"Layer {layer_idx} • {name} • Params={total:,} • Sample n={sample.size:,}"
        _ = plot_hist_and_qq_on_axes(sample, bins, title, ax_h, ax_q)

    fig_m.suptitle(f"MLP submodules — Layer {layer_idx}", fontsize=14)
    out_m = os.path.join(subdir, f"layer_{layer_idx:02d}_mlp_submodules.png")
    fig_m.savefig(out_m, dpi=150)
    plt.close(fig_m)
    print(f"[Layer {layer_idx}] Saved {out_m}")


def compute_submodule_metrics_for_layers(
    infos: Sequence[AwqTensorInfo],
    layers: List[int],
    sub_k: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Compute submodule metrics (no plotting) for given layers across all submodules.
    Returns a list of dict rows for CSV.
    """
    rows: List[Dict] = []
    for li in tqdm(layers, desc="Submodule metrics"):
        for name in ALL_SUBMODULES:
            pattern = ATTN_PATTERNS[name] if name in ATTN_PATTERNS else MLP_PATTERNS[name]
            sample, total = sample_submodule_from_awq(infos, li, pattern, sub_k, rng)
            if total == 0 or sample.size == 0:
                continue
            x = np.sort(sample.astype(np.float64))
            mu = np.mean(x); sigma = np.std(x, ddof=1)
            if not np.isfinite(sigma) or sigma == 0:
                continue
            z = (x - mu) / sigma
            rn, rl, rlg, an, al, alg = compute_quantile_metrics(z)
            best_map = {"Normal": rn, "Laplace": rl, "Logistic": rlg}
            best_family = min(best_map, key=best_map.get)
            rows.append({
                "layer": li,
                "submodule": name,
                "params": total,
                "sample_n": int(sample.size),
                "rmse_normal": rn,
                "rmse_laplace": rl,
                "rmse_logistic": rlg,
                "mae_normal": an,
                "mae_laplace": al,
                "mae_logistic": alg,
                "best_family": best_family,
            })
    return rows


# ----------------------------
# Torchinfo / fallback summaries
# ----------------------------
def write_torchinfo_or_fallback_summary(
    model_id: str,
    revision: Optional[str],
    outdir: str,
    depth: int,
    awq_infos: Sequence[AwqTensorInfo],
) -> None:
    os.makedirs(outdir, exist_ok=True)
    ok = False
    if torchinfo_summary is not None and AutoConfig is not None and AutoModelForCausalLM is not None:
        try:
            cfg = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
            if init_empty_weights is not None:
                from contextlib import nullcontext
                ctx = init_empty_weights() if init_empty_weights is not None else nullcontext()
                with ctx:
                    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
                try:
                    summ = torchinfo_summary(
                        model,
                        depth=depth,
                        verbose=0,
                        col_names=("kernel_size", "num_params", "trainable"),
                        row_settings=("var_names",)
                    )
                    outp = os.path.join(outdir, "torchinfo_summary.txt")
                    with open(outp, "w") as fh:
                        fh.write(str(summ) + "\n")
                    print(f"[torchinfo] Wrote {outp}")
                    ok = True
                except Exception as e:
                    print(f"[torchinfo] Summary failed on meta model (will fallback): {e}", file=sys.stderr)
            else:
                print("[torchinfo] accelerate not available; cannot init on meta safely; skipping torchinfo.", file=sys.stderr)
        except Exception as e:
            print(f"[torchinfo] Could not prepare config/model: {e}", file=sys.stderr)

    if ok:
        return

    # Fallback: param counts grouped by prefix using AWQ tensors
    prefix_tree: Dict[str, int] = {}
    total_params = 0
    for info in awq_infos:
        total_params += info.numel
        parts = info.full_name.split(".")
        grp = ".".join(parts[:3]) if len(parts) >= 3 else info.full_name
        prefix_tree[grp] = prefix_tree.get(grp, 0) + info.numel

    lines = [
        "AWQ tensor structural summary (fallback — torchinfo unavailable):",
        f"Total parameters (from AWQ tensors): {total_params:,}",
        "", "Top-level groups (approx.):",
    ]
    for grp, cnt in sorted(prefix_tree.items(), key=lambda x: (x[0].count("."), x[0],)):
        lines.append(f"{grp:<40} {cnt:,}")

    outp = os.path.join(outdir, "awq_tensor_summary.txt")
    with open(outp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[fallback] Wrote {outp}")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # Default outdir by arch if not provided
    if not args.outdir or args.outdir.strip() == "":
        args.outdir = "./qwen_weight_plots" if args.arch == "qwen" else "./llama3_weight_plots"

    rng = np.random.default_rng(args.seed)

    awq_dir = os.path.abspath(args.awq_dir)
    awq_infos = load_awq_tensor_infos(awq_dir)
    print(f"Using {len(awq_infos)} AWQ tensor file(s) from: {awq_dir}")
    print(f"Architecture preset: {args.arch}  |  Output dir: {args.outdir}")

    # Infer #layers from AWQ tensor metadata
    if awq_infos:
        max_layer = max(info.layer_idx for info in awq_infos)
        num_layers = max_layer + 1
    else:
        num_layers = 0
    print(f"Found {num_layers} transformer layers (by scanning AWQ tensor names).")

    # Do we need per-layer iteration?
    need_layer_iter = any([
        args.do_layer_overall_plots,
        args.do_layer_metrics_csv,
        args.do_metrics_vs_layer_plot,
        args.do_summary,
    ])

    rmse_norm_all, rmse_lap_all, rmse_log_all = [], [], []
    mae_norm_all, mae_lap_all, mae_log_all = [], [], []
    layer_indices: List[int] = []
    counted_layers = 0
    per_layer_rows = []

    # ---- Optional per-layer metrics & overall plots ----
    if need_layer_iter:
        for i in tqdm(range(num_layers), desc="Processing layers"):
            sample, n_params = sample_layer_uniform_from_awq(awq_infos, i, args.sample_per_layer, rng)
            if sample.size == 0 or n_params == 0:
                print(f"[Layer {i}] No weights found in AWQ tensors; skipping.")
                continue

            # Plot or just compute metrics depending on flag
            if args.do_layer_overall_plots:
                rn, rl, rlg, an, al, alg = make_hist_and_qq(i, sample, args.bins, args.outdir, n_params)
            else:
                # compute metrics silently
                x = np.sort(sample.astype(np.float64))
                mu = np.mean(x)
                sigma = np.std(x, ddof=1)
                if not np.isfinite(sigma) or sigma == 0:
                    print(f"[Layer {i}] Degenerate variance; skipping metrics.", file=sys.stderr)
                    continue
                z = (x - mu) / sigma
                rn, rl, rlg, an, al, alg = compute_quantile_metrics(z)
                print(f"[Layer {i:02d}] RMSE N={rn:.4f} L={rl:.4f} G={rlg:.4f} | MAE N={an:.4f} L={al:.4f} G={alg:.4f}")

            if all(np.isfinite(v) for v in (rn, rl, rlg, an, al, alg)):
                rmse_norm_all.append(rn); rmse_lap_all.append(rl); rmse_log_all.append(rlg)
                mae_norm_all.append(an);  mae_lap_all.append(al);  mae_log_all.append(alg)
                layer_indices.append(i)
                counted_layers += 1

                if args.do_layer_metrics_csv:
                    best_map = {"Normal": rn, "Laplace": rl, "Logistic": rlg}
                    best_family = min(best_map, key=best_map.get)
                    per_layer_rows.append({
                        "layer": i,
                        "params": n_params,
                        "sample_n": int(sample.size),
                        "rmse_normal": rn,
                        "rmse_laplace": rl,
                        "rmse_logistic": rlg,
                        "mae_normal": an,
                        "mae_laplace": al,
                        "mae_logistic": alg,
                        "best_family": best_family,
                    })

        # Metrics vs layer plot
        if args.do_metrics_vs_layer_plot and counted_layers > 0:
            os.makedirs(args.outdir, exist_ok=True)
            fig = plt.figure(figsize=(16, 8), constrained_layout=True)
            gs = fig.add_gridspec(2, 1)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(layer_indices, rmse_norm_all, label="RMSE Normal", linewidth=1.5)
            ax1.plot(layer_indices, rmse_lap_all,  label="RMSE Laplace", linewidth=1.5)
            ax1.plot(layer_indices, rmse_log_all,  label="RMSE Logistic", linewidth=1.5)
            ax1.set_title("Quantile RMSE vs Layer Index")
            ax1.set_xlabel("Layer index")
            ax1.set_ylabel("RMSE")
            ax1.legend(loc="best", frameon=True)

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(layer_indices, mae_norm_all, label="MAE Normal", linewidth=1.5)
            ax2.plot(layer_indices, mae_lap_all,  label="MAE Laplace", linewidth=1.5)
            ax2.plot(layer_indices, mae_log_all,  label="MAE Logistic", linewidth=1.5)
            ax2.set_title("Quantile MAE vs Layer Index")
            ax2.set_xlabel("Layer index")
            ax2.set_ylabel("MAE")
            ax2.legend(loc="best", frameon=True)

            outfile_vs = os.path.join(args.outdir, "metrics_vs_layer.png")
            fig.savefig(outfile_vs, dpi=150)
            plt.close(fig)
            print(f"Saved {outfile_vs}")

        # CSV
        if args.do_layer_metrics_csv and counted_layers > 0:
            csv_path = os.path.join(args.outdir, "layer_metrics.csv")
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "layer", "params", "sample_n",
                        "rmse_normal", "rmse_laplace", "rmse_logistic",
                        "mae_normal", "mae_laplace", "mae_logistic",
                        "best_family",
                    ],
                )
                writer.writeheader()
                for row in per_layer_rows:
                    writer.writerow(row)
            print(f"Wrote per-layer CSV: {csv_path}")

        # Text summary
        if args.do_summary:
            if counted_layers == 0:
                print("No layers produced valid samples; skipped summary.")
            else:
                def mean_std(arr):
                    arr = np.asarray(arr, dtype=np.float64)
                    if arr.size == 0:
                        return (float("nan"), float("nan"))
                    if arr.size == 1:
                        return (float(arr[0]), 0.0)
                    return (float(np.mean(arr)), float(np.std(arr, ddof=1)))

                mN, sN = mean_std(rmse_norm_all)
                mL, sL = mean_std(rmse_lap_all)
                mG, sG = mean_std(rmse_log_all)
                aN, aNs = mean_std(mae_norm_all)
                aL, aLs = mean_std(mae_lap_all)
                aG, aGs = mean_std(mae_log_all)

                os.makedirs(args.outdir, exist_ok=True)
                summary_path = os.path.join(args.outdir, "rmse_mae_summary.txt")
                with open(summary_path, "w") as fh:
                    fh.write("==== Reproducibility ====\n")
                    fh.write(f"timestamp_utc: {datetime.now(UTC).isoformat()}\n")
                    fh.write(f"model_id: {args.model_id}\n")
                    fh.write(f"revision: {args.revision}\n")
                    fh.write(f"sample_per_layer: {args.sample_per_layer}\n")
                    fh.write(f"bins: {args.bins}\n")
                    fh.write(f"seed: {args.seed}\n\n")

                    fh.write(f"Layers counted: {counted_layers}/{num_layers}\n\n")
                    fh.write("RMSE (quantile-space):\n")
                    fh.write(f"  Normal  : mean={mN:.6f}, std={sN:.6f}\n")
                    fh.write(f"  Laplace : mean={mL:.6f}, std={sL:.6f}\n")
                    fh.write(f"  Logistic: mean={mG:.6f}, std={sG:.6f}\n\n")
                    fh.write("MAE (quantile-space):\n")
                    fh.write(f"  Normal  : mean={aN:.6f}, std={aNs:.6f}\n")
                    fh.write(f"  Laplace : mean={aL:.6f}, std={aLs:.6f}\n")
                    fh.write(f"  Logistic: mean={aG:.6f}, std={aGs:.6f}\n\n")
                    fh.write("Artifacts (presence depends on flags):\n")
                    fh.write("  - Per-layer histograms and Q–Q: layer_{##}_weights.png\n")
                    fh.write("  - RMSE/MAE vs Layer: metrics_vs_layer.png\n")
                    fh.write("  - Per-layer metrics CSV: layer_metrics.csv\n")
                    fh.write("  - Submodule figures: submodules/layer_##_attn_submodules.png, submodules/layer_##_mlp_submodules.png\n")
                    fh.write("  - Submodule metrics CSV: submodule_metrics.csv\n")
                print(f"Summary written to {summary_path}")

    # ---- Submodule plots (independent) ----
    if args.do_submodule_plots:
        try:
            sub_layers = [int(s.strip()) for s in args.sub_layers.split(",") if s.strip() != ""]
        except Exception:
            print(f"WARNING: Failed to parse --sub_layers='{args.sub_layers}', using defaults [0,8,16,24,31].")
            sub_layers = [0, 8, 16, 24, 31]
        sub_layers = [i for i in sub_layers if 0 <= i < num_layers]
        if not sub_layers:
            print("No valid sub_layers within model depth; skipping submodule plots.")
        else:
            for li in sub_layers:
                plot_submodules_for_layer(awq_infos, li, args.bins, args.submodule_sample, rng, args.outdir)

    # ---- NEW: Submodule metrics CSV ----
    if args.do_submodule_metrics_csv:
        if args.submodule_metrics_all:
            target_layers = list(range(num_layers))
        else:
            try:
                target_layers = [int(s.strip()) for s in args.sub_layers.split(",") if s.strip() != ""]
            except Exception:
                print(f"WARNING: Failed to parse --sub_layers='{args.sub_layers}', using defaults [0,8,16,24,31].")
                target_layers = [0, 8, 16, 24, 31]
            target_layers = [i for i in target_layers if 0 <= i < num_layers]

        if not target_layers:
            print("No valid layers for submodule metrics; skipping CSV.")
        else:
            rows = compute_submodule_metrics_for_layers(
                infos=awq_infos,
                layers=target_layers,
                sub_k=args.submodule_sample,
                rng=rng,
            )
            if rows:
                os.makedirs(args.outdir, exist_ok=True)
                csv_path = os.path.join(args.outdir, "submodule_metrics.csv")
                with open(csv_path, "w", newline="") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=[
                            "layer", "submodule", "params", "sample_n",
                            "rmse_normal", "rmse_laplace", "rmse_logistic",
                            "mae_normal", "mae_laplace", "mae_logistic",
                            "best_family",
                        ],
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                print(f"Wrote submodule metrics CSV: {csv_path}")
            else:
                print("No submodule metrics were computed (no matching weights?).")

    # ---- Torchinfo (optional) ----
    if args.do_torchinfo:
        write_torchinfo_or_fallback_summary(
            model_id=args.model_id,
            revision=args.revision,
            outdir=args.outdir,
            depth=args.torchinfo_depth,
            awq_infos=awq_infos,
        )

    print("Done.")


if __name__ == "__main__":
    main()
