#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_fp16.h>
#include <algorithm>


#define THREADS_PER_BLOCK 1024
#define THREADS_PER_SM 1024
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 1024

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, half *data1, half *data2, half *data3, half *data4, half *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	half s2 = data2[gid];
	half s4 = data4[gid];
	half2 mult = __halves2half2(s2, s4);
	half result1 = data1[gid];
	half result2 = data3[gid];
	half2 result = __halves2half2(result1, result2);

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
		result = result*mult+result;
	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = __high2half(result) + __low2half(result);
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	half *data1 = (half*) malloc(TOTAL_THREADS*sizeof(half));
	half *data2 = (half*) malloc(TOTAL_THREADS*sizeof(half));
	half *res = (half*) malloc(TOTAL_THREADS*sizeof(half));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	half *data1_g;
	half *data2_g;
	half *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (half)i;
		data2[i] = (half)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(half)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(half)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(half)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(half), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(half), cudaMemcpyHostToDevice) );

	max_flops<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g, data1_g, data2_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(half), cudaMemcpyDeviceToHost) );


	auto dstart = *std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
	auto dend = *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS]);
	auto total_time = dend - dstart;

	float flops;
	flops = (float)(REPEAT_TIMES*THREADS_PER_SM*4)/((float)total_time);
	printf("FLOP per SM = %f (flop/clk/SM)\n", flops);
	printf("Total Clk number = %u \n", total_time);

	return 0;
} 

