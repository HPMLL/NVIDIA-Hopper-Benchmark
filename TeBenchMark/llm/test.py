"""Benchmark offline inference throughput."""
import torch
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import torch.nn as nn
import transformer_engine.pytorch as te
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
from tqdm import tqdm
# import transformers.models.llama.modeling_llama.LlamaRMSNormm as LlamaRMSNorm
import transformers
# from accelerate import Accelerator



# from vllm import LLM, SamplingParams
# from vllm.transformers_utils.tokenizer import get_tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    output_len_max: int,
    input_len_max: int
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    print("tokenizering ...")
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    print("finish tokenize !")
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if output_len_max is not None:
            output_len = output_len_max
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        if input_len_max is not None:
            prompt_len = input_len_max
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
    output_len_max: int,
    input_len_max: int,
    data_type: str
) -> float:
    assert not use_beam_search
    print("loading model", model)

    if data_type == "fp32":
        torch_type = torch.float32
    elif data_type == "fp16" or data_type == "fp8":
        torch_type = torch.bfloat16

    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch_type, trust_remote_code=trust_remote_code)
    
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token

    if data_type == "fp8":
        for name, module in list(llm.named_modules()):
            # print(name)
            
            if isinstance(module, nn.Linear):
                print("convert to te.linear")
                if any(p % 16 != 0 for p in module.weight.shape):
                    return
                has_bias = module.bias is not None
                te_module = te.Linear(
                    module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
                )
                te_module.weight.data = module.weight.data.clone()
                if has_bias:
                    te_module.bias.data = module.bias.data.clone()

                setattr(llm, name, te_module)
            # elif isinstance(module, transformers.models.llama.modeling_llama.LlamaRMSNorm):
            #     print("convert to te.RMSNorm")
                
            #     te_module = te.RMSNorm(
            #         module.weight.shape[0], eps=module.variance_epsilon, params_dtype=module.weight.dtype
            #     )
            #     te_module.weight.data = module.weight.data.clone()
            #     # te_module.eps.data = module.variance_epsilon

            #     setattr(llm, name, te_module)

    llm = llm.cuda()
    
    print("model created!!", llm.dtype, llm.device)

    pbar = tqdm(total=len(requests))
    start = time.time()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0

    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)

        if len(batch) < max_batch_size and i != len(requests) - 1:
            continue


        padded_length = (max_prompt_len + max_output_len + 15) // 16 * 16
        if input_len_max is not None:
            if padded_length > input_len_max:
                padded_length = input_len_max

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding='max_length', max_length=padded_length, truncation=True).input_ids

        # 
        # print("input", input_ids.shape)
        if output_len_max is not None:
            max_output_len = output_len_max
        with torch.autocast(device_type="cuda", dtype=torch_type):
            with te.fp8_autocast(enabled= data_type == "fp8"):
                llm_outputs = llm.generate(
                    input_ids=input_ids.cuda(),
                    do_sample=not use_beam_search,
                    num_return_sequences=n,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    max_new_tokens=max_output_len,
                )
        # print("llm_outputs", llm_outputs.shape)
        # Include the decoding time.
        llm_outputs = tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))
        # print("llm_outputs", llm_outputs)
        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.time()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.output_len, args.input_len)

    if args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code, args.output_len, args.input_len, args.dtype)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf"],
                        default="hf")
    parser.add_argument("--dtype",
                        type=str,
                        default="bf16",
                        help="Data Type.")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--output_len",
                        type=int,
                        default=128,
                        help="output_len")
    parser.add_argument("--input_len",
                        type=int,
                        default=128,
                        help="input_len")
                        
    args = parser.parse_args()

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)
