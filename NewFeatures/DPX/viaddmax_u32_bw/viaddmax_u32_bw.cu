#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
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


template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1, T *data2, T *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register T s1 = data1[gid];
	register T s2 = data2[gid]+1;
	register T s3 = s1 + 2;
	register T s4 = s2;
	register T s5 = data1[gid];
	register T s6 = data2[gid]+1;
	register T result = 1;

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {

 		s1 = __viaddmax_u32(s1, s2, s3);
 		s2 = __viaddmax_u32(s2, s3, s4);
 		s3 = __viaddmax_u32(s3, s4, s5);
 		s4 = __viaddmax_u32(s4, s5, s6);

	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	result = s1 + s2 + s3 + s4;
	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = result;
}


int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *data1 = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *data2 = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *res = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	uint32_t *data1_g;
	uint32_t *data2_g;
	uint32_t *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (uint32_t)i;
		data2[i] = (uint32_t)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(uint32_t)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyHostToDevice) );

	max_flops<uint32_t><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );

	auto dstart = *std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
	auto dend = *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS]);
	auto total_time = dend - dstart;

	float flops;
	flops = (float)(REPEAT_TIMES*THREADS_PER_SM*4)/((float)total_time);
	printf("Instructions per SM = %f (instructions/clk/SM)\n", flops);
	printf("Total Clk number = %u \n", total_time);

	return 0;
} 

