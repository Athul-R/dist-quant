
import torch
import torch.nn as nn
from .quantizer import pseudo_quantize_tensor
import gc

__all__ = ["auto_alpha_block"]

@torch.no_grad()
def auto_alpha_layer(
    w, input_feat, n_bit, q_config, n_sample_token=512
):
    # w: [co, ci]
    # input_feat: [n_token, ci]
    
    # 1. Reshape w to groups
    # [co, 1, n_group, group_size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    w_groups = w.reshape(w.shape[0], 1, -1, group_size)
    
    # 2. Reshape input_feat to groups
    # [1, n_token, n_group, group_size]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    
    # Subsample tokens if needed to save memory/time
    if n_sample_token < input_feat.shape[1]:
        input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    
    # Process in batches of output channels to avoid OOM
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
    w_all = w_groups
    best_alpha_all = []
    
    # Alpha candidates: 0.0 (Logistic) to 1.0 (Uniform)
    alpha_candidates = [x/20 for x in range(21)] # [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for i_b in range(w_groups.shape[0] // oc_batch_size):
        w_batch = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
        
        # [co_batch, 1, n_group, 1]
        best_loss = torch.full(
            (w_batch.shape[0], 1, w_batch.shape[2], 1),
            float("inf"),
            device=w.device,
            dtype=torch.float32,
        )
        best_alpha = torch.zeros_like(best_loss)
        
        input_feat_gpu = input_feat.to(w.device).float()
        
        # Compute original output for comparison? 
        # Actually min ||(w - q)x||^2 is same as min ||wx - qx||^2
        # So we can just compare error directly.
        # But to be precise let's match auto_clip style:
        # auto_clip computes err = (cur_out - org_out)^2
        
        org_out = (input_feat_gpu * w_batch).sum(dim=-1).float() # [co, n_token, n_group]
        
        for alpha in alpha_candidates:
            # Quantize with this alpha
            # pseudo_quantize_tensor_hybrid needs to be called with specific alpha
            # We assume pseudo_quantize_tensor accepts 'alpha' argument now.
            
            # Note: pseudo_quantize_tensor expects [co, ci] or similar. 
            # w_batch is [co, 1, n_group, group_size]. Reshape to 2D for quantizer?
            # pseudo_quantize_tensor handles reshaping if q_group_size is passed.
            # But here we manually reshaped. Let's pass the 4D tensor and let quantizer handle it 
            # or reshape back to 2D. 
            # auto_clip passes 4D tensor `cur_w` to `pseudo_quantize_tensor`. 
            # `pseudo_quantize_tensor` calls `_reshape_for_group_quantization`.
            # If input is already grouped [..., group_size], `_reshape` might fail if we pass q_group_size!
            # Let's check `_reshape_for_group_quantization` in quantizer.py.
            # It checks `w.shape[-1] % q_group_size == 0`.
            # If w is 4D [co, 1, ng, gs], it might be treated as [dim0, dim1*dim2*dim3].
            # Safest is to reshape to [co_batch, -1] representing [co, ci] before passing.
            
            w_batch_flat = w_batch.reshape(w_batch.shape[0], -1)
            q_config["quant_method"] = "hybrid"
            q_config["alpha"] = alpha
            q_w_flat = pseudo_quantize_tensor(
                w_batch_flat, 
                n_bit=n_bit, 
                quant_method="hybrid"
            )
            
            q_w = q_w_flat.reshape(w_batch.shape)
            
            cur_out = (input_feat_gpu * q_w).sum(dim=-1).float()
            
            # Error: [co, 1, n_group, 1]
            err = (cur_out - org_out).pow(2).mean(dim=1).view(best_loss.shape)
            
            # Update best
            improved = err < best_loss
            best_loss[improved] = err[improved]
            best_alpha[improved] = alpha
            
            del q_w_flat
            del q_w
            del cur_out
            
        best_alpha_all.append(best_alpha)
        
    best_alpha = torch.cat(best_alpha_all, dim=0)
    del input_feat
    del input_feat_gpu
    del w_groups
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_alpha.squeeze() # [co, n_group]

@torch.no_grad()
def auto_alpha_block(module, w_bit, q_config, input_feat):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }
    
    alpha_list = []
    for name in named_linears:
        # Skip QKV if needed? AWQ usually skips them for CLIP but maybe not for Alpha?
        # auto_clip skips them "due to qk bmm".
        # Let's stick to auto_clip logic for safety, or ask?
        # User said "Refer to @[awq/quantize/auto_clip.py]". 
        # Auto clip skips: if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]): continue
        # If we skip, we default to something (maybe 0.0 or 0.5).
        # But skipping means we don't optimize them.
        # Let's SKIP for now to avoid breaking things, consistent with auto_clip.
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
             continue
        
        named_linears[name].cuda()
        
        # alpha shape: [co, n_group]
        alpha = auto_alpha_layer(
            named_linears[name].weight, 
            input_feat[name], 
            n_bit=w_bit, 
            q_config=q_config
        )
        alpha_list.append((name, alpha.cpu()))
        named_linears[name].cpu()
        
    return alpha_list
