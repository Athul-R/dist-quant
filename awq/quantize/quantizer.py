import torch
import torch.nn as nn
from tqdm import tqdm
import gc
import numpy as np
import time
from scipy.stats import logistic as scipy_logistic
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name
from torch import Tensor

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
    

def logistic_fit_torch(z: Tensor, dim=None, eps: float = 1e-5):
    """
    Fit a logistic distribution to z using SciPy's maximum-likelihood estimator.
    Keeps the requested dimension for broadcasting.
    """
    device, dtype = z.device, z.dtype
    z_np = z.detach().cpu().numpy()

    if dim is None:
        loc, scale = scipy_logistic.fit(z_np.reshape(-1))
        scale = max(scale, eps)
        mu = torch.tensor(loc, device=device, dtype=dtype)
        s = torch.tensor(scale, device=device, dtype=dtype)
        return mu, s

    dim = dim if dim >= 0 else z.dim() + dim
    arr = np.moveaxis(z_np, dim, -1)
    flat = arr.reshape(-1, arr.shape[-1])
    locs = np.empty(flat.shape[0], dtype=flat.dtype)
    scales = np.empty(flat.shape[0], dtype=flat.dtype)

    for idx in range(flat.shape[0]):
        loc, scale = scipy_logistic.fit(flat[idx])
        locs[idx] = loc
        scales[idx] = max(scale, eps)

    reduced_shape = arr.shape[:-1]
    locs = locs.reshape(reduced_shape)
    scales = scales.reshape(reduced_shape)
    locs = np.expand_dims(locs, axis=dim)
    scales = np.expand_dims(scales, axis=dim)

    mu = torch.from_numpy(locs).to(device=device, dtype=dtype)
    s = torch.from_numpy(scales).to(device=device, dtype=dtype)
    return mu, s


def logistic_cdf(x, loc, scale):
    """
    Torch implementation of the Logistic CDF:
      F(x) = 1 / (1 + exp(-(x - mu) / s))
    """
    # Note: torch.sigmoid(z) is equivalent to 1 / (1 + exp(-z))
    return torch.sigmoid((x - loc) / scale)


def logistic_ppf(u, loc, scale):
    """
    Torch implementation of the Logistic PPF (inverse CDF), also known as the Logit function:
      F⁻¹(u) = mu + s * ln(u / (1 - u))
             = mu + s * (ln(u) - ln(1 - u))
    """
    # We use (log(u) - log(1-u)) for better numerical stability than log(u/(1-u))
    # Alternatively, you can use torch.logit(u) if using a recent PyTorch version.
    return loc + scale * (torch.log(u) - torch.log(1 - u))
   

def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    w_org = w

    num_levels = 2 ** n_bit - 1
    eps = 1.0 / (num_levels * 100)  # e.g., ~7.8e-5 for 7 bits

    # max_val = w.max()
    # min_val = w.min()
    # delta_prob = (max_val - min_val).clamp(min=1e-5) / (2**n_bit - 1)
    # assert torch.isfinite(delta_prob).all()

    # Z = w_org / delta_prob
    # EPS = 1e-3
    max_logistic_fit_samples = 10**5
    if w.numel() > max_logistic_fit_samples:
        flat_w = w.reshape(-1)
        sample_idx = torch.randperm(flat_w.numel(), device=w.device)[
            :max_logistic_fit_samples
        ]
        w_fit = flat_w[sample_idx]
    else:
        w_fit = w

    start = time.perf_counter()
    mu, s = logistic_fit_torch(w_fit)  # shapes (C,1), (C,1)

    w = logistic_cdf(w, loc=mu, scale=s)




    # print(f"logistic_fit_torch took {(time.perf_counter() - start) * 1000:.3f} ms")

    # n_bit_prob = n_bit
    # L = (2 ** n_bit_prob) - 1
    # uq_levels = (torch.arange(L, device=Z.device, dtype=Z.dtype) + 0.5) / L
    # uq_levels = torch.clamp(uq_levels, EPS, 1.0 - EPS)



    # start = time.perf_counter()
    # quantized_levels = logistic_decompand(uq_levels, mu, s)


    # start = time.perf_counter()
    # boundaries = (quantized_levels[:-1] + quantized_levels[1:]) / 2
    # flat_Z = Z.reshape(-1)
    # bin_idx = torch.bucketize(flat_Z, boundaries)
    # Z_hat = quantized_levels[bin_idx].reshape_as(Z)

    # w = Z_hat * delta_prob
    # assert torch.isfinite(w).all()

    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        delta = (max_val - min_val).clamp(min=eps, max=1-eps) / max_int
        zeros = (-torch.round(min_val / delta)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=eps, max=1-eps)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        delta = max_val / max_int
        zeros = 0


    assert torch.isnan(delta).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(delta).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(delta)
    else:
        w = (
            torch.clamp(torch.round(w / delta) + zeros, min_int, max_int) - zeros
        ) * delta
    assert torch.isnan(w).sum() == 0

    # Clamp to the valid (0,1) range before applying the logit/PPF. Quantization
    # can push values to exact 0/1 which would otherwise lead to inf/nan.
    w = torch.clamp(w, eps, 1 - eps)
    delta = torch.clamp(delta, eps, 1 - eps)
    w = logistic_ppf(w, loc=mu, scale=s)
    delta = logistic_ppf(delta, loc=mu, scale=s)
    w = w.reshape(org_w_shape)

    end = time.perf_counter()
    print(f"Total time taken is: {end - start}")

    # # print(f"W: {w}")
    # print("\n\nMSE >>>>")
    # mse = ((w_org - w) ** 2).mean()
    # print(mse)
    # orig_mag = w_org.abs().mean()
    # quant_mag = w.abs().mean()
    # print("Mean |w_org|:", orig_mag.item())
    # print("Mean |w|:", quant_mag.item())
    # print("|w_org| / |w| =", (orig_mag / quant_mag).item())

    
    if get_scale_zp:
        return w, delta.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
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
