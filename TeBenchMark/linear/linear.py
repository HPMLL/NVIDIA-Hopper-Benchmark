import torch
import torch.nn as nn
import time
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
# Define the model with a linear layer
class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# Define a function for benchmarking
def benchmark_linear_layer(data_type, input_size, output_size, num_samples, device):
    if data_type == "FP8":
        fp8_enable = True
        model = te.Linear(input_size, output_size, bias=True)
        input_data = torch.randn(num_samples, input_size).to(device)
    else:
        fp8_enable = False
        model = nn.Linear(input_size, output_size, bias=True)
        model = model.to(device)
        model = model.to(data_type)
        input_data = torch.randn(num_samples, input_size, device=device, dtype=data_type)
    # model = LinearModel(input_size, output_size)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Create an FP8 recipe. Note: All input args are optional.
    # fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
    # Warm-up
    torch.cuda.synchronize()
    
    
    for _ in range(10):
        with te.fp8_autocast(enabled=fp8_enable):
            _ = model(input_data)

    torch.cuda.synchronize()
    
    # Timing runs
    start.record()
    
    for _ in range(100):
        with te.fp8_autocast(enabled=fp8_enable):
            output = model(input_data)
    
    torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()
    average_inference_time = start.elapsed_time(end) / 100

    # Calculate total FLOPs
    total_flops = num_samples * output_size * input_size * 2
    
    # Calculate throughput in GFLOPs
    throughput_gflops = total_flops / (average_inference_time * 1e-3) / 1e9
    return average_inference_time, throughput_gflops

if __name__ == "__main__":
    # M_list = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    # N_list = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    # K_list = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]

    M_list = [128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 32768]
    N_list = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]
    K_list = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_types = [torch.float16, torch.float32]
    data_types = [torch.float16, torch.float32, "FP8"]
    for data_type in data_types:
        print(f"========= Data Type: {data_type} =========")
        for i in range(len(N_list)):
            # M = M_list[i]
            # N = N_list[i]
            # K = K_list[i] 
            K = N = N_list[i]
            for m in M_list:
                try:
                    avg_time, throughput_gflops = benchmark_linear_layer(data_type, K, N, m, device)
                    print(f"M: {m}, K: {K}, N: {N}, Average Inference Time: {avg_time:.6f} ms, throughput_gflops: {throughput_gflops} GFLOPs")
                except Exception as e:
                    print(e)
                    pass

    
