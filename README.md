# dist-quant
Distribution Aware Companding Quantization for transformers.

This is the repo for distribution aware companding quatization. 
The project report can be found [here](https://drive.google.com/file/d/1tnjG-KCm7NHFayHIuHcdyUgkk4Z_GNkd/view?usp=sharing).

In short we are building upon the Activating Aware Quantization ([AWQ](https://github.com/mit-han-lab/llm-awq)) and further using the weight distribution probabilities to quantize all the weights. 

The models we use are [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [LLama 3.1 1B, 3B](https://www.llama.com/models/llama-3/) models.



### Using Soft link

1. Clone `llm-awq` and `dist-quant` into your NLP folder. 


Your `dist-quant` and `llm-awq` must be in the same folder. For instance, when you `ls`, it should show both llm-awq and dist-quant

```
> ls

dist-quant
HPC Setup
llm-awq
```

Use the following command to list all the soft links.
```
ls -l

```

The output will be like the one below

```
> ls -l
total 32
-rw-r--r--@ 1 athul  staff  11357 Nov  3 12:45 LICENSE
lrwxr-xr-x@ 1 athul  staff     10 Nov  3 12:45 llm-awq -> ../llm-awq
-rw-r--r--@ 1 athul  staff    877 Nov  3 12:56 README.md
```


2. Then go to `dist-quant`. And run `cd llm-awq`. It will go to the `llm-awq` folder. 
