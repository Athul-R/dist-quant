import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import gc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .qmodule import ScaledActivation
from ..utils.module import get_op_name, set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor_uniform(
    w,
    n_bit=8,
    zero_point=True,
    q_group_size=-1,
    inplace=False,
    get_scale_zp=False,
    global_q_group=False,
    **kwargs,
):
    org_w_shape = w.shape
    if global_q_group:
        w = w.reshape(1, -1)
    elif q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    num_groups = w.shape[0]
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(num_groups, -1), zeros.view(num_groups, -1)
    else:
        return w


def _reshape_for_group_quantization(w, q_group_size, global_q_group=False):
    org_shape = w.shape
    if global_q_group:
        w_2d = w.reshape(1, -1)
    elif q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        w_2d = w.reshape(-1, q_group_size)
    else:
        w_2d = w.reshape(w.shape[0], -1)
    assert w_2d.dim() == 2
    return w_2d, org_shape


def _restore_quantized_weights(w, w_q_2d, q_group_size, org_shape, inplace):
    if inplace:
        if q_group_size > 0:
            w.view(-1, q_group_size).copy_(w_q_2d)
        else:
            w.view(w.shape[0], -1).copy_(w_q_2d)
        return w
    else:
        return w_q_2d.reshape(org_shape)


def _assign_to_codebook(w_2d, centers):
    diff = w_2d.unsqueeze(-1) - centers.unsqueeze(1)
    idx = diff.abs().argmin(dim=-1)
    w_q_2d = centers.gather(1, idx)
    return w_q_2d, idx


def _assert_finite(tensor):
    assert torch.isnan(tensor).sum() == 0
    assert torch.isfinite(tensor).all()


def _select_group_indices(total_groups, max_groups, salient_counts=None):
    max_groups = max(1, int(max_groups))
    if total_groups <= max_groups:
        indices = list(range(total_groups))
    else:
        step = total_groups / max_groups
        indices = []
        for i in range(max_groups):
            idx = min(total_groups - 1, int(round(i * step)))
            if idx not in indices:
                indices.append(idx)
        if len(indices) < max_groups:
            # backfill with trailing indices if rounding produced duplicates
            candidate = total_groups - 1
            while len(indices) < max_groups and candidate >= 0:
                if candidate not in indices:
                    indices.append(candidate)
                candidate -= 1
    if salient_counts is not None:
        salient_counts = np.asarray(salient_counts).flatten()
        if salient_counts.shape[0] == total_groups:
            ranked = np.argsort(-salient_counts)
            for idx in ranked:
                if salient_counts[idx] <= 0:
                    break
                if idx not in indices:
                    indices.append(int(idx))
                if len(indices) >= max_groups:
                    break
    return sorted(indices)


def _build_scale_lookup(scale_records):
    lookup = {}
    if not scale_records:
        return lookup
    for record in scale_records:
        if not isinstance(record, (list, tuple)) or len(record) < 3:
            continue
        _, layer_names, scales = record
        if scales is None:
            continue
        if not isinstance(layer_names, (list, tuple)):
            layer_names = (layer_names,)
        for name in layer_names:
            lookup[name] = scales
    return lookup


def _build_salient_group_mask(
    weight_tensor,
    column_scales,
    q_group_size,
    global_q_group=False,
    top_ratio=0.1,
):
    if column_scales is None or weight_tensor is None:
        return None
    if top_ratio <= 0:
        return None
    weight_last_dim = weight_tensor.shape[-1]
    if not torch.is_tensor(column_scales):
        column_scales = torch.as_tensor(column_scales)
    scales_flat = column_scales.detach().float().view(-1).cpu()
    if scales_flat.numel() == 0 or scales_flat.numel() != weight_last_dim:
        return None
    num_salient = max(1, int(math.ceil(scales_flat.numel() * top_ratio)))
    topk = torch.topk(scales_flat, num_salient, largest=True)
    salient_columns = torch.zeros_like(scales_flat, dtype=torch.bool)
    salient_columns.scatter_(0, topk.indices, True)
    salient_columns = salient_columns.to(weight_tensor.device)
    # Broadcast column selection across leading dimensions
    while salient_columns.dim() < weight_tensor.dim():
        salient_columns = salient_columns.unsqueeze(0)
    expand_shape = tuple(weight_tensor.shape)
    salient_mask = salient_columns.expand(expand_shape).to(torch.bool)
    salient_mask_2d, _ = _reshape_for_group_quantization(
        salient_mask, q_group_size, global_q_group=global_q_group
    )
    return salient_mask_2d


def _compute_weight_scale_product(weight_tensor, column_scales):
    if weight_tensor is None or column_scales is None:
        return None
    if not torch.is_tensor(column_scales):
        column_scales = torch.as_tensor(column_scales)
    scales = column_scales.detach().to(weight_tensor.dtype).view(-1)
    if scales.numel() != weight_tensor.shape[-1]:
        return None
    view_shape = [1] * (weight_tensor.dim() - 1) + [weight_tensor.shape[-1]]
    scales = scales.view(*view_shape)
    return weight_tensor * scales


def _render_histogram_with_salient(
    ax,
    values,
    salient_values,
    bins=80,
    base_color="steelblue",
    salient_color="crimson",
):
    if values is None or values.size == 0:
        return
    total_counts, bin_edges = np.histogram(values, bins=bins)
    total = max(1, total_counts.sum())
    widths = np.diff(bin_edges)
    base_counts = total_counts.astype(np.int64)
    salient_freq = None
    if salient_values is not None and salient_values.size > 0:
        salient_counts, _ = np.histogram(salient_values, bins=bin_edges)
        salient_counts = np.minimum(salient_counts, base_counts)
        base_counts = base_counts - salient_counts
        salient_freq = salient_counts.astype(np.float32) / total
    base_freq = base_counts.astype(np.float32) / total
    ax.bar(
        bin_edges[:-1],
        base_freq,
        width=widths,
        align="edge",
        color=base_color,
        alpha=0.75,
        label="Other weights",
    )
    if salient_freq is not None and np.any(salient_freq):
        ax.bar(
            bin_edges[:-1],
            salient_freq,
            width=widths,
            align="edge",
            bottom=base_freq,
            color=salient_color,
            alpha=0.9,
            label="Top 10% scale",
        )
        ax.legend(frameon=False, fontsize=8)


def _plot_weight_distributions(
    w_before,
    w_after,
    q_group_size,
    layer_idx,
    module_name,
    max_groups,
    output_dir,
    global_q_group=False,
    salient_mask=None,
    scaled_weight=None,
):
    os.makedirs(output_dir, exist_ok=True)
    w_before_2d, _ = _reshape_for_group_quantization(
        w_before, q_group_size, global_q_group=global_q_group
    )
    w_after_2d, _ = _reshape_for_group_quantization(
        w_after, q_group_size, global_q_group=global_q_group
    )
    w_scaled_2d = None
    if scaled_weight is not None:
        w_scaled_2d, _ = _reshape_for_group_quantization(
            scaled_weight, q_group_size, global_q_group=global_q_group
        )
    total_groups = w_before_2d.shape[0]
    salient_counts = None
    if salient_mask is not None:
        salient_counts = (
            salient_mask.to(torch.float32).sum(dim=1).detach().cpu().numpy()
        )
    group_indices = _select_group_indices(
        total_groups, max_groups, salient_counts=salient_counts
    )
    safe_module_name = module_name.replace(".", "_")

    for group_idx in group_indices:
        orig_vals = (
            w_before_2d[group_idx].detach().to(torch.float32).cpu().numpy().ravel()
        )
        quant_vals = (
            w_after_2d[group_idx].detach().to(torch.float32).cpu().numpy().ravel()
        )
        diff = quant_vals - orig_vals
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        max_err = float(np.max(np.abs(diff)))
        # cosine similarity between original and quantized vectors
        orig_vec = torch.from_numpy(orig_vals)
        quant_vec = torch.from_numpy(quant_vals)
        if orig_vec.norm().item() < 1e-12 or quant_vec.norm().item() < 1e-12:
            cos_sim = float("nan")
        else:
            cos_sim = F.cosine_similarity(
                orig_vec.view(1, -1), quant_vec.view(1, -1), dim=1
            ).item()
        range_ratio = float(
            (orig_vals.max() - orig_vals.min())
            / (quant_vals.max() - quant_vals.min() + 1e-12)
        )
        num_axes = 2 + (1 if w_scaled_2d is not None else 0)
        fig, axes = plt.subplots(1, num_axes, figsize=(6 * num_axes, 4))
        if num_axes == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = list(axes.reshape(-1))
        group_salient_mask = None
        if salient_mask is not None and group_idx < salient_mask.shape[0]:
            group_salient_mask = (
                salient_mask[group_idx]
                .detach()
                .to(torch.bool)
                .cpu()
                .numpy()
                .astype(bool)
            )
            if group_salient_mask.shape[0] != orig_vals.shape[0]:
                group_salient_mask = None
        salient_orig = None
        if group_salient_mask is not None and np.any(group_salient_mask):
            salient_orig = orig_vals[group_salient_mask]
        _render_histogram_with_salient(
            axes[0],
            orig_vals,
            salient_orig,
            base_color="steelblue",
            salient_color="crimson",
        )
        axes[0].set_title("Weight distribution (original)")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Normalized frequency")
        salient_quant = None
        if group_salient_mask is not None and np.any(group_salient_mask):
            salient_quant = quant_vals[group_salient_mask]
        _render_histogram_with_salient(
            axes[1],
            quant_vals,
            salient_quant,
            base_color="darkorange",
            salient_color="darkred",
        )
        axes[1].set_title("Weight distribution (quantized)")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Normalized frequency")
        if w_scaled_2d is not None:
            scaled_vals = (
                w_scaled_2d[group_idx].detach().to(torch.float32).cpu().numpy().ravel()
            )
            salient_scaled = None
            if (
                group_salient_mask is not None
                and np.any(group_salient_mask)
                and scaled_vals.shape[0] == group_salient_mask.shape[0]
            ):
                salient_scaled = scaled_vals[group_salient_mask]
            _render_histogram_with_salient(
                axes[2],
                scaled_vals,
                salient_scaled,
                base_color="seagreen",
                salient_color="darkgreen",
            )
            axes[2].set_title("Weight distribution (w × scale)")
            axes[2].set_xlabel("Value")
            axes[2].set_ylabel("Normalized frequency")
        metrics_text = (
            f"MAE: {mae:.3e}\n"
            f"MSE: {mse:.3e}\n"
            f"Max |err|: {max_err:.3e}\n"
            f"Cos sim: {cos_sim:.4f}\n"
            f"Range ratio: {range_ratio:.3f}"
        )
        axes[min(1, len(axes) - 1)].text(
            0.02,
            0.95,
            metrics_text,
            transform=axes[min(1, len(axes) - 1)].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        fig.suptitle(
            f"Layer {layer_idx} · {module_name} · Group {group_idx}"
        )
        fig.tight_layout()
        plot_name = (
            f"layer{layer_idx}_module{safe_module_name}_group{group_idx}.png"
        )
        fig.savefig(os.path.join(output_dir, plot_name), dpi=200, bbox_inches="tight")
        plt.close(fig)


def pseudo_quantize_tensor_normal(
    w,
    n_bit=8,
    q_group_size=-1,
    inplace=False,
    get_codebook=False,
    zero_point=True,
    debug=False,
    debug_prefix=None,
    get_scale_zp=False,
    codebook_spread=1.0,
    global_q_group=False,
    **kwargs
):
    """
    Non-uniform quantization using a Gaussian-based codebook.

    - Works similarly to pseudo_quantize_tensor, but instead of a linear
      scale+zero-point, it builds a per-row codebook of quantization levels.
    - Codebook centers are chosen using the inverse CDF of a normal distribution
      with row-wise mean and variance (so bins are denser near the mean).

    Args:
        w:            weight tensor (last dim is grouped if q_group_size > 0)
        n_bit:        number of bits (number of levels = 2 ** n_bit)
        q_group_size: group size along the last dimension; if > 0, we reshape
                      to (-1, q_group_size) just like in the AWQ code.
        inplace:      if True, writes the quantized values back into `w`
        get_codebook: if True, also returns the codebook (centers) per row

    Returns:
        If get_codebook is False:
            w_q          (quantized tensor, same shape as input)
        If get_codebook is True:
            w_q, centers
            - w_q:      quantized tensor, same shape as input
            - centers:  (num_rows, num_levels) codebook per row
    """

    if get_scale_zp:
        raise NotImplementedError(
            "Non-uniform pseudo quantization does not expose (scale, zero_point). "
            "Use pseudo_quantize_tensor_old for kernels that require them."
        )

    w_2d, org_shape = _reshape_for_group_quantization(
        w, q_group_size, global_q_group=global_q_group
    )
    num_rows, _ = w_2d.shape

    num_levels = 2 ** n_bit
    device = w.device
    dtype = w.dtype

    # ---- 1. Row-wise mean and variance ----
    # (You can make this per-tensor by removing dim=1 and keepdim=True if you want.)
    mean = w_2d.mean(dim=1, keepdim=True)                     # (num_rows, 1)
    var = w_2d.var(dim=1, unbiased=False, keepdim=True)       # (num_rows, 1)
    std = var.clamp(min=1e-5).sqrt()                          # avoid zero-variance

    # ---- 2. Build Gaussian-based quantization centers (codebook) ----
    # We put the centers at Gaussian quantiles:
    # p_k = (k + 0.5) / num_levels  for k=0,...,L-1
    # z_k = Phi^{-1}(p_k)  (standard normal)
    # center_k = mean + std * z_k
    #
    # Using erfinv: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)

    working_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    k = torch.arange(num_levels, device=device, dtype=working_dtype)          # (L,)
    p = (k + 0.5) / num_levels                                                # (L,), in (0,1)
    # Standard normal quantiles
    z = math.sqrt(2.0) * torch.erfinv(2 * p - 1)                              # (L,)
    if z.dtype != dtype:
        z = z.to(dtype)

    # Expand to per-row codebook: centers shape (num_rows, num_levels)
    centers = mean + (std * codebook_spread) * z.unsqueeze(0)

    # ---- 3. Assign each weight to nearest center (non-uniform quantization) ----
    # w_2d:      (num_rows, row_dim)
    # centers:   (num_rows, num_levels)
    # We want nearest center along the "levels" axis.
    #
    # diff: (num_rows, row_dim, num_levels)
    w_q_2d, idx = _assign_to_codebook(w_2d, centers)

    if debug:
        _log_non_uniform_quant_stats(
            debug_prefix or "pseudo_quantize_tensor", w_2d, w_q_2d, centers, idx
        )

    w_q = _restore_quantized_weights(w, w_q_2d, q_group_size, org_shape, inplace)
    _assert_finite(w_q)

    if get_codebook:
        # reshape centers to match "rows" after grouping
        return w_q, centers
    else:
        return w_q


def pseudo_quantize_tensor_logistic(
    w,
    n_bit=8,
    q_group_size=-1,
    inplace=False,
    get_codebook=False,
    zero_point=True,
    debug=False,
    debug_prefix=None,
    get_scale_zp=False,
    codebook_spread=1.0,
    global_q_group=False,
    **kwargs,
):
    """Non-uniform quantization using a logistic distribution derived codebook."""

    if get_scale_zp:
        raise NotImplementedError(
            "Logistic pseudo quantization does not expose (scale, zero_point)."
        )

    w_2d, org_shape = _reshape_for_group_quantization(
        w, q_group_size, global_q_group=global_q_group
    )
    num_levels = 2 ** n_bit
    device = w.device
    dtype = w.dtype

    mean = w_2d.mean(dim=1, keepdim=True)
    var = w_2d.var(dim=1, unbiased=False, keepdim=True)
    std = var.clamp(min=1e-5).sqrt()
    spread_std = std * codebook_spread
    logistic_scale = (spread_std * math.sqrt(3.0) / math.pi).clamp(min=1e-5)

    working_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    k = torch.arange(num_levels, device=device, dtype=working_dtype)
    p = (k + 0.5) / num_levels
    z = torch.logit(p, eps=1e-6)
    if z.dtype != dtype:
        z = z.to(dtype)

    centers = mean + logistic_scale * z.unsqueeze(0)
    w_q_2d, idx = _assign_to_codebook(w_2d, centers)

    if debug:
        _log_non_uniform_quant_stats(
            debug_prefix or "pseudo_quantize_tensor_logistic",
            w_2d,
            w_q_2d,
            centers,
            idx,
        )

    w_q = _restore_quantized_weights(w, w_q_2d, q_group_size, org_shape, inplace)
    _assert_finite(w_q)

    if get_codebook:
        return w_q, centers
    else:
        return w_q



def pseudo_quantize_tensor_hybrid(
    w,
    n_bit=8,
    q_group_size=-1,
    inplace=False,
    get_codebook=False,
    zero_point=True,
    debug=False,
    debug_prefix=None,
    get_scale_zp=False,
    codebook_spread=1.0,
    global_q_group=False,
    alpha=None,
    **kwargs,
):
    """
    Hybrid Uniform-Logistic quantization.
    Mixes Logistic (density-matching) and Uniform (range-matching) centroids.
    
    If `alpha` is None, performs a grid search to find the best alpha per-group.
    Alpha 0.0 = Pure Logistic
    Alpha 1.0 = Pure Uniform
    """

    if get_scale_zp:
        raise NotImplementedError(
            "Hybrid pseudo quantization does not expose (scale, zero_point)."
        )

    w_2d, org_shape = _reshape_for_group_quantization(
        w, q_group_size, global_q_group=global_q_group
    )
    # w_2d: (num_rows, row_dim)
    
    num_levels = 2 ** n_bit
    device = w.device
    dtype = w.dtype
    working_dtype = (
        torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    )

    # --- 1. Compute Base Statistics ---
    # Logistic components
    mean = w_2d.mean(dim=1, keepdim=True)
    var = w_2d.var(dim=1, unbiased=False, keepdim=True)
    std = var.clamp(min=1e-5).sqrt()
    spread_std = std * codebook_spread
    logistic_scale = (spread_std * math.sqrt(3.0) / math.pi).clamp(min=1e-5)
    
    k = torch.arange(num_levels, device=device, dtype=working_dtype)
    p = (k + 0.5) / num_levels
    z_logistic = torch.logit(p, eps=1e-6)
    if z_logistic.dtype != dtype:
        z_logistic = z_logistic.to(dtype)
    
    # Centers if pure Logistic
    # shape: (num_rows, num_levels)
    centers_logistic = mean + logistic_scale * z_logistic.unsqueeze(0)

    # Uniform components
    # We want max_val such that [-max_val, max_val] covers the range?
    # Or just min/max? Uniform usually means linear spacing between min(w) and max(w).
    # Let's use simple min/max of the row.
    row_min = w_2d.amin(dim=1, keepdim=True)
    row_max = w_2d.amax(dim=1, keepdim=True)
    
    # Linear spacing from min to max
    # step = (max - min) / (2^n - 1)
    # levels = min + step * k
    step = (row_max - row_min).clamp(min=1e-5) / (num_levels - 1)
    k_idx = torch.arange(num_levels, device=device, dtype=dtype) # Use weight dtype
    centers_uniform = row_min + step * k_idx.unsqueeze(0)

    alpha_val = alpha if alpha is not None else 0.5 # Default to 0.5 if not found
    
    # If alpha_val is a tensor, we need to broadcast/reshape it to match (num_rows, num_levels)
    # alpha_val might be (co, n_group) or (num_rows,)
    # w_2d is (num_rows, row_dim)
    # centers is (num_rows, num_levels)
    
    if torch.is_tensor(alpha_val):
        alpha_val = alpha_val.to(device).to(working_dtype)
        # Assuming alpha_val is (num_rows,) or broadcastable to (num_rows, 1)
        if alpha_val.dim() == 1 and alpha_val.numel() == w_2d.shape[0]:
             alpha_val = alpha_val.unsqueeze(1) # (num_rows, 1)
        # If it was (co, ngroup), it should have been reshaped before passing here?
        # Ideally, passed alpha should match the grouping of w_2d.
        
    best_centers = (1 - alpha_val) * centers_logistic + alpha_val * centers_uniform

    # --- 3. Final Quantization ---
    w_q_2d, idx = _assign_to_codebook(w_2d, best_centers)

    if debug:
        _log_non_uniform_quant_stats(
            debug_prefix or "pseudo_quantize_tensor_hybrid",
            w_2d,
            w_q_2d,
            best_centers,
            idx,
        )

    w_q = _restore_quantized_weights(w, w_q_2d, q_group_size, org_shape, inplace)
    _assert_finite(w_q)

    if get_codebook:
        return w_q, best_centers
    else:
        return w_q


def pseudo_quantize_tensor(
    w,
    *args,
    quant_method="uniform",
    **kwargs,
):
    """Dispatch to the requested pseudo quantization routine."""

    method = kwargs.pop("quant_method", quant_method)
    method = (method or "uniform").lower()

    if kwargs.get("get_scale_zp") and method != "uniform":
        raise NotImplementedError(
            "Only uniform quantization supports (scale, zero_point) outputs."
        )

    if method == "uniform":
        return pseudo_quantize_tensor_uniform(w, *args, **kwargs)
    if method in ("normal", "gaussian"):
        return pseudo_quantize_tensor_normal(w, *args, **kwargs)
    if method == "logistic":
        return pseudo_quantize_tensor_logistic(w, *args, **kwargs)
    if method in ("hybrid", "saliency", "saliency_aware"):
        return pseudo_quantize_tensor_hybrid(w, *args, **kwargs)

    raise ValueError(f"Unsupported quantization method: {method}")


def _log_non_uniform_quant_stats(prefix, w_orig, w_quant, centers, idx):
    """Print a small collection of stats that make debugging easier."""
    diff = (w_quant - w_orig).abs()
    mean_abs_err = diff.mean().item()
    max_abs_err = diff.max().item()
    cos = F.cosine_similarity(
        w_orig.reshape(1, -1).float(), w_quant.reshape(1, -1).float(), dim=1
    ).item()

    row_range = w_orig.max(dim=1).values - w_orig.min(dim=1).values
    codebook_range = centers.max(dim=1).values - centers.min(dim=1).values
    range_ratio = (row_range / (codebook_range + 1e-6)).mean().item()

    num_levels = centers.shape[1]
    idx_equal_min = (idx == 0).float()
    idx_equal_max = (idx == num_levels - 1).float()
    clip_low = idx_equal_min.mean().item()
    clip_high = idx_equal_max.mean().item()
    heavy_clip_rows = (
        (idx_equal_min.mean(dim=1) > 0.25) | (idx_equal_max.mean(dim=1) > 0.25)
    ).float()

    print(
        f"[quant-debug] {prefix}: mean|err|={mean_abs_err:.4e}, "
        f"max|err|={max_abs_err:.4e}, cos={cos:.4f}, "
        f"range_ratio={range_ratio:.2f}"
    )
    print(
        f"[quant-debug] {prefix}: clip_low={clip_low * 100:.2f}%, "
        f"clip_high={clip_high * 100:.2f}%, "
        f"rows>25%clipped={heavy_clip_rows.mean().item() * 100:.2f}%"
    )



def _build_alpha_lookup(alpha_records):
    lookup = {}
    if not alpha_records:
        return lookup
    for record in alpha_records:
        if not isinstance(record, (list, tuple)) or len(record) < 2:
            continue
        _, layer_names, alphas = record
        if alphas is None:
            continue
        if not isinstance(layer_names, (list, tuple)):
            layer_names = (layer_names,)
        for name in layer_names:
            lookup[name] = alphas
    return lookup


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
    awq_scale_records=None,
    awq_alpha_records=None,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    debug_enabled = q_config.get("debug", False)
    plot_enabled = q_config.get("plot_quant_dists", False)
    max_plot_groups = q_config.get("plot_quant_groups", 4)
    plot_dir = q_config.get("plot_quant_dir", "quant_plots")
    
    awq_scale_lookup = _build_scale_lookup(awq_scale_records)
    awq_alpha_lookup = _build_alpha_lookup(awq_alpha_records)
    
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        layer_prefix = get_op_name(model, layers[i])
        if layer_prefix:
            layer_prefix = layer_prefix + "."
        for n, m in named_linears.items():
            full_module_name = f"{layer_prefix}{n}" if layer_prefix else n
            module_scales = awq_scale_lookup.get(full_module_name)
            module_alphas = awq_alpha_lookup.get(full_module_name)
            
            m.cuda()
            extra_kwargs = {}
            if debug_enabled:
                extra_kwargs["debug_prefix"] = f"layer_{i}.{n}"
            
            # Pass alpha if available
            if module_alphas is not None:
                # module_alphas: [co, n_group]
                # We need to flatten it to match w_groups: [co * n_group] or [co, n_group] ?
                # pseudo_quantize_tensor_hybrid reshapes w to (num_rows, -1). 
                # If q_group_size > 0: num_rows = co * (ci / group_size)
                # module_alphas shape is [co, ci / group_size].
                # So flattening it to (-1) should align with rows of w_2d.
                extra_kwargs["alpha"] = module_alphas.view(-1)
                
            w_before_cpu = None
            w_before_scaled_cpu = None
            if plot_enabled and i == 0:
                w_before_cpu = m.weight.data.detach().cpu().clone()
                if module_scales is not None:
                    w_before_scaled_cpu = _compute_weight_scale_product(
                        w_before_cpu, module_scales
                    )
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config, **extra_kwargs
            )
            if w_before_cpu is not None:
                try:
                    w_after_cpu = m.weight.data.detach().cpu().clone()
                    salient_mask = None
                    if module_scales is not None:
                        salient_mask = _build_salient_group_mask(
                            w_before_cpu,
                            module_scales,
                            q_config.get("q_group_size", -1),
                            global_q_group=q_config.get("global_q_group", False),
                            top_ratio=0.1,
                        )
                    _plot_weight_distributions(
                        w_before_cpu,
                        w_after_cpu,
                        q_config.get("q_group_size", -1),
                        layer_idx=i,
                        module_name=n,
                        max_groups=max_plot_groups,
                        output_dir=plot_dir,
                        global_q_group=q_config.get("global_q_group", False),
                        salient_mask=salient_mask,
                        scaled_weight=w_before_scaled_cpu,
                    )
                    del w_after_cpu
                except Exception as exc:
                    print(
                        f"[quant-plot] Failed to save plot for layer {i} {n}: {exc}"
                    )
                finally:
                    del w_before_cpu
                    if w_before_scaled_cpu is not None:
                        del w_before_scaled_cpu
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
