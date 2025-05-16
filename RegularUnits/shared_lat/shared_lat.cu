//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the latency of L1 cache

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>


#define SHARED_MEM_SIZE_BYTE (48*1024) //size in bytes, max 96KB for v100
#define SHARED_MEM_SIZE (SHARED_MEM_SIZE_BYTE/8)
#define THREADS_NUM 32   //Launch only one thread to calcaulte the latency using a pointer-chasing array technique
#define WARP_SIZE 32
#define ITERS 2048       //iterate over the array ITERS times

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

//TO DO: @Jason, please change the code to be similar to the L2/DRAM latency format
//Measure latency of ITERS reads. 
__global__ void shared_lat(uint32_t *startClk, uint32_t *stopClk, uint64_t *dsink, uint32_t stride){

	// thread index
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid*blockDim.x+tid;
    uint32_t n_threads = blockDim.x * gridDim.x;
    
    __shared__ uint64_t s[SHARED_MEM_SIZE]; //static shared memory

	// one thread to initialize the pointer-chasing array
	for (uint32_t i=uid; i<(SHARED_MEM_SIZE-stride); i+=n_threads)
            s[i] = (i+stride)%SHARED_MEM_SIZE;

	if(uid == 0){
        //initalize pointer chaser
		uint64_t p_chaser = 0;
	
		// start timing
		uint32_t start = 0;
		asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");


		// pointer-chasing ITERS times
		// use ca modifier to cache the load in L1
		for(uint32_t i=0; i<ITERS; ++i) {	
			p_chaser = s[p_chaser];
		}

		// stop timing
		uint32_t stop = 0;
		asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

		// write time and data back to memory
		startClk[uid] = start;
		stopClk[uid] = stop;
		dsink[uid] = p_chaser;
	}
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(sizeof(uint32_t));
	uint64_t *dsink = (uint64_t*) malloc(sizeof(uint64_t));
	
	uint32_t *startClk_g;
    uint32_t *stopClk_g;
    uint64_t *dsink_g;
	
	gpuErrchk( cudaMalloc(&startClk_g, sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&dsink_g, sizeof(uint64_t)) );
	
	shared_lat<<<1,THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g, 1);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	printf("Shared Memory Latency  = %f cycles\n", (float)(stopClk[0]-startClk[0])/ITERS );
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);
	printf("start clk = %u \n", startClk[0]);
	printf("stop clk = %u \n", stopClk[0]);

	return 0;
} 
