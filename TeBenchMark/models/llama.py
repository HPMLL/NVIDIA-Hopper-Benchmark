# from utils import DotProductAttention, speedometer
import utils
import torch
import transformer_engine.pytorch as te
import argparse

import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='benchmark')

parser.add_argument('--model', default="llama", type=str)
parser.add_argument('--dtype', default="fp8", type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--sequence_length', default=512, type=int)
parser.add_argument('--hidden_size', default=1024, type=int)
parser.add_argument('--num_attention_heads', default=16, type=int)
parser.add_argument('--ffn_hidden_size', default=4096, type=int)
parser.add_argument('--n_layers', default=1, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(1234)
    # Layer configuration
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    # ffn_hidden_size = args.ffn_hidden_size

    multiple_of = 256
    ffn_hidden_size = int((2 * hidden_size * 4) / 3)
    ffn_hidden_size = ((ffn_hidden_size - 1) // multiple_of) * multiple_of + multiple_of

    n_layers = args.n_layers
    batch_size = args.batch_size
    sequence_length = args.sequence_length

    fp8_enable = False
    if args.dtype == "fp8":
        fp8_enable = True
        dtype = torch.float16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Synthetic data
    x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
    dy = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)

    
    layers = [te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads,
        num_gqa_groups=None,
        bias=False,
        activation='swiglu',
        normalization='RMSNorm',
        self_attn_mask_type='causal', 
        layer_number=1,layernorm_epsilon=1e-6,
        layer_type='encoder',
        qkv_weight_interleaved=True, 
        hidden_dropout=0.0, attention_dropout=0.0,
        fuse_qkv_params=True) 
        for _ in range(n_layers)]

    model = torch.nn.Sequential(
        *layers
    )
    model.to(dtype=dtype).cuda()

    print(f"model: {args.model}, hidden_size: {args.hidden_size}, num_attention_heads: {args.num_attention_heads}, n_layers: {args.n_layers} dtype: {args.dtype}, batch_size: {batch_size}, sequence_length: {sequence_length}")
    utils.speedometer(
        model,
        x,
        dy,
        # forward_kwargs = { "attention_mask": None },
        fp8_autocast_kwargs = { "enabled": fp8_enable}
    )
