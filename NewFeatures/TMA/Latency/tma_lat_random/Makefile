NVCC = /usr/local/cuda/bin/nvcc

# internal flags
NVCCFLAGS   :=  -std=c++11 -arch=sm_90 
CCFLAGS     := 
LDFLAGS     := -lcublas -lnvidia-ml
NAME 		:= latency
PREFIX		:= .
N 			:= 1


$(PREFIX)/cuda-$(NAME): main.cu Makefile gpu-clock.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)/hip-$(NAME): main.hip Makefile ../rocm-metrics/rocm-metrics.hpp
	echo $(HIP_HOME)
	hipcc -std=c++20 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl -o $@ $<

clean:
	rm -f cuda-$(NAME) hip-$(NAME)
