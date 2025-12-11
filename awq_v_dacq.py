import argparse
import csv
from pathlib import Path

import torch
from torch import Tensor

EPS = 1e-4


def logistic_fit_torch(z: Tensor, dim=None, eps: float = EPS):
    """
    Fit a logistic distribution to z along given dim.
    Logistic: mean = mu, var = (pi^2 / 3) * s^2  =>  s = std * sqrt(3) / pi
    Returns (mu, s) with dims kept for broadcasting.
    """
    mu = z.mean(dim=dim, keepdim=True)
    std = z.std(dim=dim, keepdim=True) + eps
    s = std * (3.0 ** 0.5) / torch.pi
    return mu, s


def logistic_cdf_torch(z: Tensor, mu: Tensor, s: Tensor, eps: float = EPS) -> Tensor:
    """
    Logistic CDF (sigmoid) with clamping to avoid exact 0/1, which would explode logit.
    """
    out = torch.sigmoid((z - mu) / s)
    return torch.clamp(out, eps, 1.0 - eps)


def logistic_icdf_torch(u: Tensor, mu: Tensor, s: Tensor, eps: float = EPS) -> Tensor:
    """
    Logistic inverse CDF (quantile):
    F^{-1}(u) = mu + s * log(u / (1 - u))
    """
    logit_u = torch.logit(u, eps=eps)
    return mu + s * logit_u


def awq_vs_dacq_logistic_torch(
    W0: Tensor,
    s_c: Tensor | float,
    delta_awq: Tensor | float,
    bit_width: int = 4,
    per_channel: bool = True,
    verbose: bool = True,
):
    """
    Compare vanilla AWQ vs DACQ (logistic companding) for a given weight matrix
    and AWQ scales, all in PyTorch.

    Args:
        W0          : Tensor of shape (C, D), raw pretrained weights.
        s_c         : scalar or Tensor of shape (C,), AWQ importance scaling factors.
        delta_awq   : scalar or Tensor of shape (C,), AWQ step sizes Δ'_c.
        bit_width   : integer bit-width (e.g., 4).
        per_channel : if True, fit logistic per channel (dim 0); else global.
        verbose     : print MSE and some stats if True.

    Returns:
        dict with:
          - 'W_ref'       : AWQ-scaled reference weights
          - 'W_awq_deq'   : dequantized vanilla AWQ weights
          - 'W_dacq_deq'  : dequantized DACQ (logistic) weights
          - 'q_awq'       : AWQ int weights
          - 'q_dacq'      : DACQ int weights
          - 'delta_awq'   : AWQ scale(s) used
          - 'delta2'      : final DACQ affine scale(s)
          - 'mse_awq'     : scalar tensor
          - 'mse_dacq'    : scalar tensor
    """
    # Ensure tensor type
    W0 = W0.to(torch.float32)
    C, D = W0.shape

    device = W0.device

    # Prepare s_c
    if not torch.is_tensor(s_c):
        s_c = torch.tensor(s_c, dtype=torch.float32, device=device)
    if s_c.ndim == 0:
        s_c = s_c.expand(C)  # (C,)
    assert s_c.shape[0] == C
    s_c = s_c.view(C, 1)  # (C,1)

    # Prepare delta_awq
    if not torch.is_tensor(delta_awq):
        delta_awq = torch.tensor(delta_awq, dtype=torch.float32, device=device)
    if delta_awq.ndim == 0:
        delta_awq = delta_awq.expand(C)  # (C,)
    assert delta_awq.shape[0] == C
    # Prevent divide-by-zero/Inf when normalizing by Δ'_c.
    delta_awq = delta_awq.view(C, 1).clamp_min(EPS)  # (C,1)

    # Signed quantization range
    Qmax = 2 ** (bit_width - 1) - 1  # e.g., 7 for 4-bit

    # ----------------------------------------------------
    # 1) AWQ-scaled reference weights: w = s_c * W0
    # ----------------------------------------------------
    W_ref = s_c * W0  # (C, D)

    # ----------------------------------------------------
    # 2) Vanilla AWQ: uniform quantization with delta_awq
    # ----------------------------------------------------
    q_awq = torch.round(W_ref / delta_awq)
    q_awq = torch.clamp(q_awq, -Qmax, Qmax)
    W_awq_deq = q_awq * delta_awq

    # ----------------------------------------------------
    # 3) DACQ: logistic companding on top of AWQ
    # ----------------------------------------------------
    # 3a) Normalize by delta_awq (Δ'_c)
    Z = W_ref / delta_awq  # (C, D)

    # 3b) Fit logistic distribution
    if per_channel:
        # Fit per row (channel)
        mu, s = logistic_fit_torch(Z, dim=1)  # shapes (C,1), (C,1)
    else:
        # Global fit
        mu, s = logistic_fit_torch(Z, dim=None)  # shapes (1,1), (1,1)

    # 3c) Companding: map Z -> U via logistic CDF
    U = logistic_cdf_torch(Z, mu, s)

    # 3d) Uniform quantization in U-space
    L = 2 ** bit_width
    Uq = (torch.floor(U * L) + 0.5) / L  # midpoint quantization
    Uq = torch.clamp(Uq, EPS, 1.0 - EPS)

    # 3e) Decompanding: Uq -> Z_hat via logistic ICDF
    Z_hat = logistic_icdf_torch(Uq, mu, s)

    # 3f) Undo normalization: back to AWQ-scaled space
    finfo = torch.finfo(Z_hat.dtype)
    Z_hat = torch.nan_to_num(Z_hat, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    W_tilde = Z_hat * delta_awq  # (C, D)
    W_tilde = torch.nan_to_num(W_tilde, nan=0.0, posinf=finfo.max, neginf=-finfo.max)

    # ----------------------------------------------------
    # 4) Final hardware-friendly affine int quantization
    # ----------------------------------------------------
    # Per-channel final scale Δ''_c from W_tilde
    max_abs = torch.max(torch.abs(W_tilde), dim=1, keepdim=True).values
    max_abs = torch.clamp(max_abs, min=EPS)
    delta2 = max_abs / Qmax  # (C,1)
    delta2 = torch.nan_to_num(delta2, nan=EPS, posinf=finfo.max, neginf=EPS)

    q_dacq = torch.round(W_tilde / delta2)
    q_dacq = torch.clamp(q_dacq, -Qmax, Qmax)
    W_dacq_deq = q_dacq * delta2

    # ----------------------------------------------------
    # 5) MSE comparison vs reference W_ref
    # ----------------------------------------------------
    mse_awq = torch.mean((W_awq_deq - W_ref) ** 2)
    mse_dacq = torch.mean((W_dacq_deq - W_ref) ** 2)

    if verbose:
        print("=== AWQ vs DACQ (logistic, torch) ===")
        print(f"bit_width          : {bit_width}")
        print(f"per_channel_fit    : {per_channel}")
        print(f"MSE (AWQ)          : {mse_awq.item():.6e}")
        print(f"MSE (DACQ-logistic): {mse_dacq.item():.6e}")
        print("delta_awq (first few):", delta_awq[:4, 0].detach().cpu().numpy())
        print("delta2    (first few):", delta2[:4, 0].detach().cpu().numpy())

    return {
        "W_ref": W_ref,
        "W_awq_deq": W_awq_deq,
        "W_dacq_deq": W_dacq_deq,
        "q_awq": q_awq.to(torch.int8),
        "q_dacq": q_dacq.to(torch.int8),
        "delta_awq": delta_awq,
        "delta2": delta2,
        "mse_awq": mse_awq,
        "mse_dacq": mse_dacq,
    }


def load_awq_tensors(weight_path: str, bit_width: int) -> tuple[Tensor, Tensor, Tensor]:
    payload = torch.load(weight_path, map_location="cpu")
    if "weights" not in payload or "awq_scale" not in payload:
        raise ValueError(f"{weight_path} must contain 'weights' and 'awq_scale' tensors.")

    weight = payload["weights"].to(torch.float32).contiguous()
    awq_scale = torch.as_tensor(payload["awq_scale"], dtype=torch.float32).view(-1)

    if awq_scale.shape[0] == weight.shape[0]:
        W0 = weight
        s_c = awq_scale
    elif awq_scale.shape[0] == weight.shape[1]:
        W0 = weight.t().contiguous()
        s_c = awq_scale
    else:
        raise ValueError(
            f"Cannot align AWQ scale (len={awq_scale.shape[0]}) with weight shape {weight.shape}."
        )

    Qmax = 2 ** (bit_width - 1) - 1
    max_abs = torch.max(torch.abs(W0), dim=1).values.clamp(min=EPS)
    delta_awq = (max_abs / Qmax).view(-1)
    return W0, s_c, delta_awq


def process_awq_directory(
    directory: str, bit_width: int, per_channel: bool, csv_out: str
) -> tuple[list[dict], tuple[float, float]]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"{directory} is not a directory.")

    weight_files = sorted(dir_path.glob("*.pt"))
    if not weight_files:
        raise ValueError(f"No .pt files found in {directory}.")

    results: list[dict] = []
    total_elements = 0
    total_awq_error = 0.0
    total_dacq_error = 0.0

    for weight_file in weight_files:
        W0, s_c, delta_awq = load_awq_tensors(str(weight_file), bit_width)
        out = awq_vs_dacq_logistic_torch(
            W0,
            s_c=s_c,
            delta_awq=delta_awq,
            bit_width=bit_width,
            per_channel=per_channel,
            verbose=False,
        )
        mse_awq = out["mse_awq"].item()
        mse_dacq = out["mse_dacq"].item()
        numel = out["W_ref"].numel()
        total_elements += numel
        total_awq_error += mse_awq * numel
        total_dacq_error += mse_dacq * numel
        stem = weight_file.stem
        winner = "DACQ" if mse_dacq < mse_awq else "AWQ"
        layer_idx = None
        parts = stem.split("_")
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                layer_idx = int(parts[2])
            except ValueError:
                layer_idx = None
        results.append(
            {
                "module": stem,
                "layer": layer_idx if layer_idx is not None else "",
                "mse_awq": mse_awq,
                "mse_dacq": mse_dacq,
                "winner": winner,
            }
        )

    combined_awq = total_awq_error / max(total_elements, 1)
    combined_dacq = total_dacq_error / max(total_elements, 1)

    fieldnames = ["module", "layer", "mse_awq", "mse_dacq", "winner"]
    with open(csv_out, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
        writer.writerow(
            {
                "module": "ALL",
                "layer": "",
                "mse_awq": combined_awq,
                "mse_dacq": combined_dacq,
                "winner": "DACQ" if combined_dacq < combined_awq else "AWQ",
            }
        )

    return results, (combined_awq, combined_dacq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AWQ vs DACQ on saved AWQ tensors.")
    parser.add_argument(
        "--awq-weight",
        type=str,
        default=None,
        help="Path to a saved AWQ weight file (produced by extract_awq_tensors.py).",
    )
    parser.add_argument(
        "--awq-dir",
        type=str,
        default=None,
        help="Directory containing multiple AWQ weight files to process in batch.",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        default=4,
        help="Bit width used for quantization (default: 4).",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Fit logistic distribution per channel (default: False unless AWQ file provided).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default="awq_vs_dacq_mse.csv",
        help="Path to save CSV report when processing a directory of AWQ weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.awq_dir:
        per_channel = True
        results, (combined_awq, combined_dacq) = process_awq_directory(
            args.awq_dir, args.bit_width, per_channel, args.csv_out
        )
        print(
            f"Processed {len(results)} modules from {args.awq_dir}; CSV saved to {args.csv_out}."
        )
        print(f"Combined MSE - AWQ: {combined_awq:.6e}, DACQ: {combined_dacq:.6e}")
        return

    if args.awq_weight:
        W0, s_c, delta_awq = load_awq_tensors(args.awq_weight, args.bit_width)
        per_channel = True
    else:
        # Example fallback
        W0 = torch.tensor([[0.10, -0.30, 0.50]], dtype=torch.float32)
        s_c = torch.tensor([2.0], dtype=torch.float32)
        delta_awq = torch.tensor([1.0 / 7.0], dtype=torch.float32)
        per_channel = args.per_channel or True

    out = awq_vs_dacq_logistic_torch(
        W0,
        s_c=s_c,
        delta_awq=delta_awq,
        bit_width=args.bit_width,
        per_channel=per_channel,
        verbose=True,
    )

    print("W_ref      :", out["W_ref"])
    print("W_awq_deq  :", out["W_awq_deq"])
    print("W_dacq_deq :", out["W_dacq_deq"])
    print("q_awq      :", out["q_awq"])
    print("q_dacq     :", out["q_dacq"])


if __name__ == "__main__":
    main()
