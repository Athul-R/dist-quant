import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

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
def pseudo_quantize_tensor_old(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
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
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


def pseudo_quantize_tensor(
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

    org_shape = w.shape

    # Handle grouping like in original AWQ pseudo_quantize_tensor
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        w_2d = w.reshape(-1, q_group_size)
    else:
        w_2d = w.reshape(w.shape[0], -1)

    assert w_2d.dim() == 2
    num_rows, row_dim = w_2d.shape

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

    k = torch.arange(num_levels, device=device, dtype=dtype)          # (L,)
    p = (k + 0.5) / num_levels                                        # (L,), in (0,1)
    # Standard normal quantiles
    z = math.sqrt(2.0) * torch.erfinv(2 * p - 1)                      # (L,)

    # Expand to per-row codebook: centers shape (num_rows, num_levels)
    centers = mean + (std * codebook_spread) * z.unsqueeze(0)

    # ---- 3. Assign each weight to nearest center (non-uniform quantization) ----
    # w_2d:      (num_rows, row_dim)
    # centers:   (num_rows, num_levels)
    # We want nearest center along the "levels" axis.
    #
    # diff: (num_rows, row_dim, num_levels)
    diff = w_2d.unsqueeze(-1) - centers.unsqueeze(1)
    idx = diff.abs().argmin(dim=-1)                      # (num_rows, row_dim), indices in [0, num_levels-1]

    # Gather quantized values from centers
    w_q_2d = centers.gather(1, idx)                      # (num_rows, row_dim)

    if debug:
        _log_non_uniform_quant_stats(
            debug_prefix or "pseudo_quantize_tensor", w_2d, w_q_2d, centers, idx
        )

    # ---- 4. Write back (in-place or not) ----
    if inplace:
        # respect the original tensor storage
        if q_group_size > 0:
            w.view(-1, q_group_size).copy_(w_q_2d)
        else:
            w.view(w.shape[0], -1).copy_(w_q_2d)
        w_q = w
    else:
        w_q = w_q_2d.reshape(org_shape)

    assert torch.isnan(w_q).sum() == 0
    assert torch.isfinite(w_q).all()

    if get_codebook:
        # reshape centers to match "rows" after grouping
        return w_q, centers
    else:
        return w_q


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



@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    debug_enabled = q_config.get("debug", False)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            extra_kwargs = {}
            if debug_enabled:
                extra_kwargs["debug_prefix"] = f"layer_{i}.{n}"
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config, **extra_kwargs
            )
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
