python -m awq.entry --model_path Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama3-8b-w4-g128-dacq.pt --codebook_spread 10 --fixed_scale

python -m awq.entry --model_path Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128-dacq.pt \
    --q_backend fake --codebook_spread 10
