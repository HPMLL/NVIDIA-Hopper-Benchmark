//This code is a modification of microbenchmarks from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of shared memory for 32 bit read

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define SHARED_MEM_SIZE_BYTE (48*1024) //size in bytes, max 96KB for v100
#define SHARED_MEM_SIZE (SHARED_MEM_SIZE_BYTE/4)
//#define SHARED_MEM_SIZE (16384)
#define ITERS (4096)

#define BLOCKS_NUM 1
#define THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void shared_bw(uint32_t *startClk, uint32_t *stopClk, uint32_t *dsink, uint32_t stride){
    
    // thread index
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid*blockDim.x+tid;
	uint32_t n_threads = blockDim.x * gridDim.x;
	
	// a register to avoid compiler optimization
	//uint32_t sink0 = 0;
	register uint32_t tmp = uid;
	
	uint32_t start = 0;
	uint32_t stop = 0;

    __shared__ uint32_t s[SHARED_MEM_SIZE]; //static shared memory
	//uint32_t s[SHARED_MEM_SIZE];
    // one thread to initialize the pointer-chasing array
	for (uint32_t i=uid; i<(SHARED_MEM_SIZE); i+=n_threads)
		s[i] = (i+stride)%SHARED_MEM_SIZE;

	// synchronize all threads
	asm volatile ("bar.sync 0;");
	
	// start timing
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// load data from l1 cache and accumulate
	for(uint32_t i=0; i<ITERS; ++i){
		tmp = s[tmp];
	}

	// synchronize all threads
	asm volatile("bar.sync 0;");
	
	// stop timing
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	//sink0 = tmp;
	// write time and data back to memory
	startClk[uid] = start;
	stopClk[uid] = stop;
	dsink[uid] = tmp;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *dsink = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	
	uint32_t *startClk_g;
    uint32_t *stopClk_g;
    uint32_t *dsink_g;
		
	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(uint32_t)) );
	
	shared_bw<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g, 1024);
    gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );

    double bw;
	bw = (double)(ITERS*TOTAL_THREADS*4)/((double)(stopClk[0]-startClk[0]));
	printf("Shared Memory Bandwidth = %f (byte/clk/SM)\n", bw);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
}