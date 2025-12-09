
import torch
import numpy as np
from awq.quantize.quantizer import pseudo_quantize_tensor

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_heavy_tailed_data(shape, outlier_prob=0.01, outlier_scale=10.0):
    # Standard normal data
    data = torch.randn(shape)
    
    # Add outliers
    mask = torch.rand(shape) < outlier_prob
    outliers = torch.randn(shape) * outlier_scale
    data = torch.where(mask, outliers, data)
    return data

def test_quantization():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Generate data: (128 rows, 1024 cols)
    # Simulates a weight matrix
    shape = (128, 1024)
    w = generate_heavy_tailed_data(shape, outlier_prob=0.01, outlier_scale=5.0).to(device)
    
    # 2. Generate Input (Calibration Data)
    input_feat = torch.randn(128, 1024).to(device) # [n_token, ci]
    
    # 3. Baseline: Logistic Quantization
    print("\n--- Testing Logistic Quantization (Baseline) ---")
    w_logistic = pseudo_quantize_tensor(
        w.clone(), 
        n_bit=4, 
        q_group_size=128, 
        quant_method="logistic"
    )
    # Metric: Activation MSE
    out_orig = input_feat @ w.t()
    out_logistic = input_feat @ w_logistic.t()
    loss_logistic = (out_orig - out_logistic).pow(2).mean().item()
    print(f"Logistic Activation MSE: {loss_logistic:.6f}")
    
    # 4. New Method: Hybrid Quantization (Auto Alpha)
    print("\n--- Testing Hybrid Quantization (Auto Alpha) ---")
    
    # We simulate auto_alpha logic here manually for verification
    best_loss = float('inf')
    best_alpha = -1
    best_w_hybrid = None
    
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for alpha in candidates:
        w_cand = pseudo_quantize_tensor(
            w.clone(), 
            n_bit=4, 
            q_group_size=128, 
            quant_method="hybrid", 
            alpha=alpha
        )
        out_cand = input_feat @ w_cand.t()
        loss = (out_orig - out_cand).pow(2).mean().item()
        print(f"  Alpha {alpha}: Loss {loss:.6f}")
        
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_w_hybrid = w_cand
            
    print(f"Best Alpha Global (Simplified): {best_alpha}")
    print(f"Hybrid Activation MSE: {best_loss:.6f}")
    
    # 5. Compare
    print("\n--- Comparison ---")
    print(f"Act MSE Improvement: {loss_logistic - best_loss:.6e} ({((loss_logistic - best_loss)/loss_logistic)*100:.2f}%)")
    
    if best_loss <= loss_logistic:
        print("\nSUCCESS: Hybrid method matched or improved Activation MSE.")
    else:
        print("\nWARNING: Hybrid method Activation MSE is worse.")

if __name__ == "__main__":
    test_quantization()
