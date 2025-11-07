from llm_awq.awq.quantize.pre_quant import get_named_linears
from transformers import AutoModelForCausalLM
import torch
import os


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

LLM_PATH = "Meta-Llama-3-8B-hf/hf"
VISUALIZATIONS_DIR = "visualizations"
AWQ_RESULTS_PATH = "llm-awq/awq_cache/llama3-8b-w4-g128.pt"


def get_salient_weights(model, awq_results):
    """
    Extract and visualize salient weights based on AWQ scale results.
    
    Parameters:
    -----------
    model : transformers model
        The loaded language model
    awq_results : dict
        AWQ results containing 'scale' and 'clip' keys
    """
    linears = get_named_linears(model)
    scales_list = awq_results["scale"]
    
    print(f"Found {len(linears)} linear layers and {len(scales_list)} scale entries\n")
    print(f"Scales list: {scales_list[0]}")


    
    for block_name, scale_names, scale_tensor in scales_list:
        print(f"\nBlock: {block_name}")
        print(f"Scale tensor shape: {scale_tensor.shape}")
        
        if scale_tensor.ndim == 0:
            print(f"Warning: Block {block_name} has 0-dimensional scale (scalar), skipping...")
            continue
        
        scale_vector = scale_tensor.detach().cpu().float().numpy()
        print(f"Scale vector shape: {scale_vector.shape}")
        
        if scale_vector.size == 0:
            print(f"Warning: Empty scale vector for block {block_name}, skipping...")
            continue
        
        for scale_name in scale_names:
            print(f"  Processing layer: {scale_name}")
            
            if scale_name not in linears:
                print(f"  Warning: {scale_name} not found in model linears, skipping...")
                continue
            
            linear_layer = linears[scale_name]
            weight_matrix = linear_layer.weight.detach().cpu().float().numpy()
            
            print(f"  Weight matrix shape: {weight_matrix.shape}")
            
            max_scale_idx = np.argmax(scale_vector)
            max_scale_value = scale_vector[max_scale_idx]
            print(f"  Max scale index: {max_scale_idx}, value: {max_scale_value:.4f}")
            
            fig = visualize_weight_matrix(weight_matrix, scale_vector, layer_name=scale_name, figsize=(14, 10))
            
            sanitized_name = scale_name.replace('.', '_').replace('/', '_')
            save_dir = VISUALIZATIONS_DIR
            os.makedirs(save_dir, exist_ok=True)
            output_file = f"{save_dir}/visualization_{sanitized_name}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {output_file}")
            plt.close(fig)


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print("Loading AWQ scales...")
    awq_results = torch.load(AWQ_RESULTS_PATH)
    
    print(f"AWQ results keys: {awq_results.keys()}")
    print(f"Number of scale entries: {len(awq_results['scale'])}")
    print(f"Number of clip entries: {len(awq_results['clip'])}\n")
    
    get_salient_weights(model, awq_results)


def visualize_weight_matrix(matrix, scale_vector, layer_name='', figsize=(12, 10), cmap='viridis'):
    """
    Visualize a weight matrix with column-wise coloring based on scale vector.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Weight matrix of shape (rows, cols)
    scale_vector : numpy.ndarray
        Vector of shape (cols,) containing magnitude values for each column
    layer_name : str
        Name of the layer being visualized
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name for the scale vector
    """
    
    rows, cols = matrix.shape
    
    max_col_idx_scale = np.argmax(scale_vector)
    max_scale_value = scale_vector[max_col_idx_scale]
    col_avg_magnitude = np.mean(np.abs(matrix), axis=0)
    max_col_idx_weight = np.argmax(col_avg_magnitude)
    
    mean_max_scale_col = np.mean(matrix[:, max_col_idx_scale])
    mean_max_weight_col = np.mean(matrix[:, max_col_idx_weight])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                     gridspec_kw={'width_ratios': [20, 1]})
    
    norm = Normalize(vmin=scale_vector.min(), vmax=scale_vector.max())
    cmap_obj = plt.cm.get_cmap(cmap)
    
    img = np.zeros((rows, cols, 3))
    for col in range(cols):
        color = cmap_obj(norm(scale_vector[col]))[:3]
        col_norm = (matrix[:, col] - matrix[:, col].min()) / (matrix[:, col].max() - matrix[:, col].min() + 1e-8)
        img[:, col, :] = col_norm[:, np.newaxis] * color
    
    ax1.imshow(img, aspect='auto', interpolation='nearest')
    
    ax1.axvline(x=max_col_idx_scale, color='red', linewidth=2, linestyle='--', 
                label=f'Max AWQ scale (col {max_col_idx_scale}, scale={max_scale_value:.4f}, mean={mean_max_scale_col:.4f})')
    ax1.axvline(x=max_col_idx_weight, color='cyan', linewidth=2, linestyle='--', 
                label=f'Max weight magnitude (col {max_col_idx_weight}, mean={mean_max_weight_col:.4f})')
    
    ax1.set_xlabel('Input Features', fontsize=12)
    ax1.set_ylabel('Output Features', fontsize=12)
    
    title = f'{layer_name}\n'
    title += f'Weight Matrix: {rows}×{cols} | '
    title += f'Scale range: [{scale_vector.min():.2f}, {scale_vector.max():.2f}] | '
    title += f'Weight range: [{matrix.min():.2f}, {matrix.max():.2f}]'
    ax1.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax2)
    cbar.set_label('AWQ Scale Magnitude', fontsize=11)
    
    ax1.text(0.02, 0.98, f'Total params: {rows * cols:,}', 
             transform=ax1.transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Example usage with synthetic data
def visualize_weight_matrix_example():
    # Create example 4096x4096 matrix (using smaller size for demo, adjust as needed)
    # For actual 4096x4096, this will work but may be slower
    size = 512  # Change to 4096 for full size
    
    # Generate random weight matrix
    np.random.seed(42)
    weight_matrix = np.random.randn(size, size)
    
    # Generate scale vector (e.g., L2 norms of columns, attention scores, etc.)
    scale_vector = np.random.rand(size) * 10
    
    # Add some structure: make certain columns have higher scales
    scale_vector[100:150] = np.random.rand(50) * 20
    scale_vector[250] = 25  # This will be highlighted as maximum
    
    # Visualize
    fig = visualize_weight_matrix(weight_matrix, scale_vector)
    plt.savefig('weight_matrix_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Matrix shape: {weight_matrix.shape}")
    print(f"Scale vector shape: {scale_vector.shape}")
    print(f"Maximum column index: {np.argmax(scale_vector)}")
    print(f"Maximum scale value: {scale_vector.max():.2f}")


if __name__ == "__main__":
    main()