import torch
import torch.nn as nn
from tqdm import tqdm
import gc
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
    Fit a logistic distribution to z along given dim.
    Logistic: mean = mu, var = (pi^2 / 3) * s^2  =>  s = std * sqrt(3) / pi
    Returns (mu, s) with dims kept for broadcasting.
    """
    mu = z.mean(dim=dim, keepdim=True)
    std = z.std(dim=dim, keepdim=True) + eps
    s = std * (3.0 ** 0.5) / torch.pi
    return mu, s


def logistic_cdf_torch(z: Tensor, mu: Tensor, s: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Logistic CDF (sigmoid) with clamping to avoid exact 0/1, which would explode logit.
    """
    out = torch.sigmoid((z - mu) / s)
    return torch.clamp(out, eps, 1.0 - eps)


def logistic_icdf_torch(u: Tensor, mu: Tensor, s: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Logistic inverse CDF (quantile):
    F^{-1}(u) = mu + s * log(u / (1 - u))
    """
    logit_u = torch.logit(u, eps=eps)
    return mu + s * logit_u

   

def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    w_org = w

    max_val = w.max()
    min_val = w.min()
    delta_prob = (max_val - min_val).clamp(min=1e-5) / (2**n_bit - 1)
    assert torch.isfinite(delta_prob).all()

    Z = w_org / delta_prob
    mu, s = logistic_fit_torch(Z, dim=1)  # shapes (C,1), (C,1)
    U = logistic_cdf_torch(Z, mu, s)

    n_bit_prob = n_bit
    EPS = 1e-3
    L = (2 ** n_bit_prob) - 1
    Uq = (torch.floor(U * L) + 0.5) / L  # midpoint quantization
    Uq = torch.clamp(Uq, EPS, 1.0 - EPS)

    Z_hat = logistic_icdf_torch(Uq, mu, s)

    w = Z_hat * delta_prob
    assert torch.isfinite(w).all()

    # if q_group_size > 0:
    #     assert org_w_shape[-1] % q_group_size == 0
    #     w = w.reshape(-1, q_group_size)
    # assert w.dim() == 2
    # if zero_point:
    #     max_val = w.amax(dim=1, keepdim=True)
    #     min_val = w.amin(dim=1, keepdim=True)
    #     max_int = 2**n_bit - 1
    #     min_int = 0
    #     delta = (max_val - min_val).clamp(min=1e-5) / max_int
    #     zeros = (-torch.round(min_val / delta)).clamp_(min_int, max_int)
    # else:  # we actually never used this
    #     assert min_val is None
    #     max_val = w.abs().amax(dim=1, keepdim=True)
    #     max_val = max_val.clamp(min=1e-5)
    #     max_int = 2 ** (n_bit - 1) - 1
    #     min_int = -(2 ** (n_bit - 1))
    #     delta = max_val / max_int
    #     zeros = 0


    #assert torch.isnan(scales).sum() == 0
    #assert torch.isnan(w).sum() == 0

    # if inplace:
    #     (
    #         (w.div_(delta).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
    #     ).mul_(delta)
    # else:
    #     w = (
    #         torch.clamp(torch.round(w / delta) + zeros, min_int, max_int) - zeros
    #     ) * delta
    # assert torch.isnan(w).sum() == 0

    # w = w.reshape(org_w_shape)

    # if get_scale_zp:
    #     return w, delta.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    # else:
    #     return w

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
