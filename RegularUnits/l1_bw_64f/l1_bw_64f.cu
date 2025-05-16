//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L1 cache for 64 bit read

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_SM 1024
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 256
#define ARRAY_SIZE 16384    //ARRAY_SIZE has to be less than L1_SIZE
#define L1_SIZE 16384  //L1 size in 64-bit. Volta L1 size is 128KB, i.e. 16K of 64-bit

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void l1_bw(uint32_t *startClk, uint32_t *stopClk, double *dsink, double *posArray){
	
	// thread index
	uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
	
	// a register to avoid compiler optimization
	double sink0 = 0;
	double sink1 = 0;
	
	// populate l1 cache to warm up
	for (uint32_t i = tid; i<ARRAY_SIZE; i+=THREADS_PER_BLOCK) {
		double* ptr = posArray + i;
		// use ca modifier to cache the load in L1
		asm volatile ("{\t\n"
			".reg .f64 data;\n\t"
			"ld.global.ca.f64 data, [%1];\n\t"
			"add.f64 %0, data, %0;\n\t"
			"}" : "+d"(sink0) : "l"(ptr) : "memory"
		);
	}
	
	// synchronize all threads
	asm volatile ("bar.sync 0;");
	
	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
	
	// load data from l1 cache and accumulate
	for(uint32_t j=0; j<REPEAT_TIMES; j++){
        	        double* ptr = posArray + ((tid + (j*WARP_SIZE*2))%ARRAY_SIZE);
                	asm volatile ("{\t\n"
                        	".reg .f64 data<2>;\n\t"
				"ld.global.ca.f64 data0, [%2+0];\n\t"
				"ld.global.ca.f64 data1, [%2+256];\n\t"
				"add.f64 %0, data0, %0;\n\t"
				"add.f64 %1, data1, %1;\n\t"
				"}" : "+d"(sink0),"+d"(sink1) : "l"(ptr) : "memory"
                	);
	}

	// synchronize all threads
	asm volatile("bar.sync 0;");
	
	// stop timing
	uint32_t stop = 0;
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	startClk[uid] = start;
	stopClk[uid] = stop;
	dsink[uid] = sink0+sink1;
}

int main(){
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	double *posArray = (double*) malloc(ARRAY_SIZE*sizeof(double));
	double *dsink = (double*) malloc(TOTAL_THREADS*sizeof(double));
	
	uint32_t *startClk_g;
        uint32_t *stopClk_g;
        double *posArray_g;
        double *dsink_g;
	
	for (uint32_t i=0; i<ARRAY_SIZE; i++)
		posArray[i] = (double)i;
		
	gpuErrchk( cudaMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(double)) );
	gpuErrchk( cudaMalloc(&dsink_g, TOTAL_THREADS*sizeof(double)) );
	
	gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(double), cudaMemcpyHostToDevice) );

	l1_bw<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g, posArray_g);
        gpuErrchk( cudaPeekAtLastError() );
	
	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, TOTAL_THREADS*sizeof(double), cudaMemcpyDeviceToHost) );

        double bw;
	bw = (double)(REPEAT_TIMES*THREADS_PER_SM*8*2)/((double)(stopClk[0]-startClk[0]));
	printf("L1 bandwidth = %f (byte/clk/SM)\n", bw);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
} 
