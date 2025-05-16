#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <algorithm>

#define THREADS_PER_BLOCK 1024
#define BLOCKS_NUM 264 //(128 4090) (108 A100) (114 H800)
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
__global__ void max_flops(uint64_t *startClk, uint64_t *stopClk, T *data1, T *data2, T *res) {
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
	uint64_t start = 0;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {

 		s1 = __vimax_s16x2_relu(s1, s2);
 		s2 = __vimax_s16x2_relu(s2, s3);
 		s3 = __vimax_s16x2_relu(s3, s4);
 		s4 = __vimax_s16x2_relu(s4, s5);

	}
	// synchronize all threads
	asm volatile("bar.sync 0;");

	// stop timing
	uint64_t stop = 0;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop));

	result = s1 + s2 + s3 + s4;
	// write time and data back to memory
	startClk[gid] = start;
	stopClk[gid] = stop;
	res[gid] = result;
}


int main(){
	uint64_t *startClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	int32_t *data1 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *data2 = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));
	int32_t *res = (int32_t*) malloc(TOTAL_THREADS*sizeof(int32_t));

	uint64_t *startClk_g;
	uint64_t *stopClk_g;
	int32_t *data1_g;
	int32_t *data2_g;
	int32_t *res_g;

	for (uint64_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (uint64_t)i;
		data2[i] = (uint64_t)i;
	}

	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(int32_t)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(int32_t)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyHostToDevice) );

	max_flops<int32_t><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(int32_t), cudaMemcpyDeviceToHost) );

	float bw;
	unsigned long long instructions = (unsigned long long)TOTAL_THREADS * REPEAT_TIMES;


	auto dstart = *std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
	auto dend = *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS]);
	auto total_time = dend - dstart;

	int dev;
	cudaDeviceProp deviceProp;
	gpuErrchk(cudaGetDevice(&dev));
	gpuErrchk(cudaGetDeviceProperties(&deviceProp, dev));

	float total_clock = (total_time) * (deviceProp.clockRate * 1e-6f);

	bw = (float)(instructions*4) / (total_clock);
	printf("DPX throughput = %f (instructions/cycle)\n", bw);
	printf("Total Clk number = %u \n", (unsigned)total_clock);


	return 0;
} 

