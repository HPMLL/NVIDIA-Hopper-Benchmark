batch_size=8

### llama-7b###

for dtype in fp32 fp16 fp8
do
    # Adjust log file name based on data type
    log_filename="./log_v2/llama-7b/${dtype}-b${batch_size}-hf-a100.log"
    
    # Construct the command string dynamically
    command="python test.py --backend hf --dataset ../ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /workspace/dudayou/hf-models/hf-llama-2-7b/ \
    --num-prompts 504 --hf-max-batch-size ${batch_size} \
    --input_len 128 --output_len 128 \
    --dtype ${dtype} \
    2>&1 | tee ${log_filename}"
    
    # Output the running command
    echo "Running command: ${command}"
    
    # Execute the
    eval "${command}"
done


# ## llama-3b fp32
# mkdir -p ./log_v2/llama-3b/
for dtype in fp32 fp16 fp8
do
     # Adjust log file name based on data type
     log_filename="./log_v2/llama-3b/${dtype}-b${batch_size}-hf-a100.log"
   
     # Construct the command string dynamically
    command="python test.py --backend hf --dataset ../ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /workspace/dudayou/hf-models/hf-llama-2-3b/ \
    --num-prompts 504 --hf-max-batch-size ${batch_size} \
    --input_len 128 --output_len 128 \
    --dtype ${dtype} \
    2>&1 | tee ${log_filename}"
   
    # Output the running command
    echo "Running command: ${command}"
    # Execute the command
    eval "${command}"
done

# ## llama-13b 
# mkdir -p ./log_v2/llama-13b/
for dtype in fp32 fp16 fp8
do
    # Adjust log file name based on data type
    log_filename="./log_v2/llama-13b/${dtype}-b${batch_size}-hf-a100.log"
   
    # Construct the command string dynamically
    command="python test.py --backend hf --dataset ../ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /workspace/dudayou/hf-models/hf-llama-2-13b/ \
    --num-prompts 504 --hf-max-batch-size ${batch_size} \
    --input_len 128 --output_len 128 \
    --dtype ${dtype} \
    2>&1 | tee ${log_filename}"
    
    # Output the running command
    echo "Running command: ${command}"
    # Execute the command
    eval "${command}"
done
