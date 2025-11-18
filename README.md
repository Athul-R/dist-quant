# dist-quant
Distribution Aware Companding Quantization for transformers.

This is the repo for distribution aware companding quatization. 
The project report can be found [here](https://drive.google.com/file/d/1tnjG-KCm7NHFayHIuHcdyUgkk4Z_GNkd/view?usp=sharing).

In short we are building upon the Activating Aware Quantization ([AWQ](https://github.com/mit-han-lab/llm-awq)) and further using the weight distribution probabilities to quantize all the weights. 

The models we use are [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [LLama 3.1 1B, 3B](https://www.llama.com/models/llama-3/) models.



2. Then go to `dist-quant`. And run `cd llm-awq`. It will go to the `llm-awq` folder.

## Extracting AWQ scales and weights

Use the `extract_awq_tensors.py` helper to dump AWQ scales, raw weights, and scale-weight products to disk for downstream visualization. The script relies on the `llm-awq` repo being available (via the soft link above) and a saved AWQ cache file.

Example usage:

```
python extract_awq_tensors.py \
  --model ./../Meta-Llama-3-8B-hf/hf \
  --awq-cache llm-awq/awq_cache/llama3-8b-w4-g128.pt \
  --output-dir weights \
  --save-weights
```

* `--model` should point to a local model directory or a Hugging Face model ID.
* `--awq-cache` is the path to the AWQ checkpoint produced by quantization.
* `--output-dir` controls where the `.npy` files and `summary.json` manifest are written.
* Add `--save-weights` if you also want the raw weight matrices (these can be large).

Each processed linear layer produces three files in the output directory:

* `<layer>_scale.npy` – the AWQ scale vector.
* `<layer>_scale_weight.npy` – elementwise product of the scale vector and the layer weight matrix.
* `<layer>_weight.npy` – the raw weight matrix (only when `--save-weights` is passed).

A `summary.json` file lists every exported layer and the paths to its artifacts for quick inspection.
