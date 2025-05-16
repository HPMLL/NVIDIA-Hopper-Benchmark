from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, AutoTokenizer
from transformer_engine.common.recipe import Format, DelayedScaling
import argparse
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, interval=99999999, amax_history_len=512, amax_compute_algo="max")
prompt = "What is microbenchmark "

from accelerate import Accelerator

# accelerator = Accelerator(mixed_precision="fp8")


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    # model = model.half()
    # model = accelerator.prepare(model)
    i = 0
    for name, module in list(model.named_modules()):
        # print(name)
        
        if isinstance(module, nn.Linear):
            i += 1
            if i > 15:
                break
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

            setattr(model, name, te_module)
    model = model.to(torch.bfloat16).cuda()
    # for name, module in model.named_modules():
    #     print(name, module)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, return_tensors="pt",
                              padding=True).input_ids



    # swith te.fp8_autocast(enabled=True, fp8_recipe= fp8_recipe):
    outputs = model.generate(
            input_ids=input_ids.cuda(),
            do_sample=False,
            num_return_sequences=1,
            temperature=1,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=512,
        )
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Prompt:", prompt)
    print("Answer:", outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    args = parser.parse_args()
    main(args)