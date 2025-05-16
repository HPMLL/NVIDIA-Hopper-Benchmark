//This code is a modification of L2 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L2 cache for 64 bit
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

//This code have been tested on Volta V100 architecture

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>
#include <cassert>

#define BLOCKS_NUM 114
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM * THREADS_NUM)
#define REPEAT_TIMES 4096//2048
#define WARP_SIZE 32 
#define ARRAY_SIZE (TOTAL_THREADS + REPEAT_TIMES*WARP_SIZE)    //Array size must not exceed L2 size 
#define ELEMENT_SIZE 8

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/

__global__ void init_data(double*dsink, double* posArray) {
	// block and thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t uid = bid * blockDim.x + tid;
	// a register to avoid compiler optimization
	double sink = 0;	
	// warm up l2 cache
	for(uint32_t i = uid; i<ARRAY_SIZE; i+=TOTAL_THREADS){
		double* ptr = posArray+i;
		// every warp loads all data in l2 cache
		// use cg modifier to cache the load in L2 and bypass L1
		asm volatile("{\t\n"
			".reg .f64 data;\n\t"
			"ld.global.cg.f64 data, [%1];\n\t"
			"add.f64 %0, data, %0;\n\t"
			"}" : "+d"(sink) : "l"(ptr) : "memory"
		);
	}
	// dsink[bid*THREADS_NUM+tid] = sink;

}

__global__ void l2_bw (uint64_t*startClk, uint64_t*stopClk, double*dsink, double*posArray){
	// block and thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t uid = bid * blockDim.x + tid;
	// a register to avoid compiler optimization
	double sink = 0;
	// start timing
	uint64_t start = 0, stop = 0;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	
	// benchmark starts
	// load data from l2 cache and accumulate,
	for(uint32_t i = 0; i<REPEAT_TIMES; i++){
			double* ptr = posArray+(i*WARP_SIZE)+uid;
			asm volatile("{\t\n"
				".reg .f64 data;\n\t"
				"ld.global.cg.f64 data, [%1];\n\t"
				"add.f64 %0, data, %0;\n\t"
				"}" : "+d"(sink) : "l"(ptr) : "memory"
			);
	}
	asm volatile("bar.sync 0;");

	// stop timing
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop));


	// store the result
	startClk[bid*THREADS_NUM+tid] = start;
	stopClk[bid*THREADS_NUM+tid] = stop;
	dsink[bid*THREADS_NUM+tid] = sink;
}

int main(){
	int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

	assert(prop.l2CacheSize / ELEMENT_SIZE > ARRAY_SIZE);


	uint64_t *startClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));
	uint64_t *stopClk = (uint64_t*) malloc(TOTAL_THREADS*sizeof(uint64_t));

	double *posArray = (double*) malloc(ARRAY_SIZE*sizeof(double));
	double *dsink = (double*) malloc(TOTAL_THREADS*sizeof(double));

	double *posArray_g;
	double *dsink_g;
	uint64_t *startClk_g;
	uint64_t *stopClk_g;

	for (int i=0; i<ARRAY_SIZE; i++)
		posArray[i] = (double)i;

	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(double)) );
	gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(double)) );
	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint64_t)) );

	gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(double), cudaMemcpyHostToDevice) );

	init_data<<<BLOCKS_NUM,THREADS_NUM>>>(dsink_g, posArray_g);
	l2_bw<<<BLOCKS_NUM,THREADS_NUM>>>(startClk_g, stopClk_g, dsink_g, posArray_g);
	gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(double), cudaMemcpyDeviceToHost) );

	float bw;
	unsigned long long data = (unsigned long long)TOTAL_THREADS*REPEAT_TIMES*8;
	uint64_t dstart = *std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
	uint64_t dend = *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS]);
	uint64_t total_time = dend - dstart;
	
	int dev;
	cudaDeviceProp deviceProp;
	gpuErrchk( cudaGetDevice(&dev));
	gpuErrchk( cudaGetDeviceProperties(&deviceProp, dev));


	float total_clock = (total_time) * (deviceProp.clockRate * 1e-6f);

	bw = (float)(data)/(total_clock);
	printf("L2 bandwidth = %f (byte/cycle)\n", bw);
	printf("Total Clk number = %u \n", (unsigned)total_clock);

	return 0;
}
