# llama

rm ./logs/llama-b4h512-layers-h800.log

hidden_sizes=(1024 2048 4096 5120 8192)
num_attention_heads=(8 16 32 40 64)
n_layers=(8 16 32 40 80)
ffn_hidden_sizes=(2816 5632 11008 13824 22016)
dtypes=("fp8" "fp16" "fp32")

for i in "${!hidden_sizes[@]}"; do
    hidden_size="${hidden_sizes[$i]}"
    num_heads="${num_attention_heads[$i]}"
    ffn_hidden_size="${ffn_hidden_sizes[$i]}"
    for dtype in "${dtypes[@]}"; do
        python llama.py                    \
            --model llama                   \
            --dtype $dtype                          \
            --batch_size 4                       \
            --sequence_length 512               \
            --hidden_size $hidden_size           \
            --num_attention_heads $num_heads     \
            --ffn_hidden_size $ffn_hidden_size                     \
            2>&1 | tee -a ./logs/llama-b4h512-layers-h800.log                     
    done
done