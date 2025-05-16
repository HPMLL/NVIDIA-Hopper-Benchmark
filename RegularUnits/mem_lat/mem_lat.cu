// This code is a modification of L1 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the latency of GPU memory

// This code have been tested on Volta V100 architecture

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//#define FINE_GRAINED
#define THREADS_NUM 1 // HERE, we launch four threads, to ensure that one request is equal to DRAM trascation, 4 thread * 8 bytes = 32 bytes (= min DRAM trascation)
#define ITERS 8192			
#define STRIDE 32 // bytes
#define ARRAY_SIZE (8*1024*128) //1048576 * 8 = 8 MB ,2621440 * 8 = 20 MB//(104857600 4090) (52428800 A100) (62914560 H800)  //pointer-chasing array size in 64-bit. total array size is 7 MB which larger than L2 cache size (6 MB in Volta) to avoid l2 cache resident from the copy engine
#define BLOCKS 114			//(128 4090) (108 A100) (114 H800)
#define THREADS_PER_BLOCK 1024
#define TOTAL_THREADS BLOCKS *THREADS_PER_BLOCK

// GPU error check
#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

__global__ void init_data(uint64_t *posArray) {
	// thread index
	uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;

	// uint32_t smid;
    // asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
	// if (tid == 0)
	// 	printf("BLOCK ID: %d, SMID: %d\n",blockIdx.x, smid);

	// initialize pointer-chasing array
	for (uint32_t i = uid; i < (ARRAY_SIZE - THREADS_NUM); i += TOTAL_THREADS)
		posArray[i] = (uint64_t)(posArray + i + STRIDE / sizeof(uint64_t)); 

	if (uid < THREADS_NUM)
	{ // only THREADS_NUM has to be active here

		// initialize the tail to reference to the head of the array
		posArray[ARRAY_SIZE - (THREADS_NUM - tid)] = (uint64_t)(posArray + tid);
	}
}

__global__ void mem_lat(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint64_t *dsink)
{
	// thread index
	uint32_t tid = threadIdx.x;

	// uint32_t smid;
    // asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
	// if (tid == 0)
	// 	printf("SMID: %d\n", smid);

	uint64_t *ptr = posArray + tid;
	uint64_t ptr1, ptr0;

	// initialize the pointers with the start address
	// Here, we use cache volatile modifier to ignore the L2 cache
	asm volatile("{\t\n"
					"ld.global.cv.u64 %0, [%1];\n\t"
					"}" : "=l"(ptr1) : "l"(ptr) : "memory");

	// synchronize all threads
	asm volatile("bar.sync 0;");

	uint32_t start = 0;
	uint32_t stop = 0;

#ifdef FINE_GRAINED
	uint32_t tem_start, tem_stop;
	__shared__ uint32_t tem[1024];
#endif

	// start timing
	asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

	// pointer-chasing ITERS times
	// Here, we use cache volatile modifier to ignore the L2 cache
	for (uint32_t i = 0; i < ITERS; i++)
	{
#ifdef FINE_GRAINED
		asm volatile("mov.u32 %0, %%clock;" : "=r"(tem_start)::"memory");
#endif
		asm volatile("{\t\n"
						"ld.global.cg.u64 %0, [%1];\n\t"
						"}" : "=l"(ptr0) : "l"((uint64_t *)ptr1) : "memory");
		ptr1 = ptr0; // swap the register for the next load

#ifdef FINE_GRAINED
		tem[tid] = ptr0;
		asm volatile("mov.u32 %0, %%clock;" : "=r"(tem_stop)::"memory");
		if (tid == 0) {
			printf("%d\n", tem_stop - tem_start);			
		}
		__syncthreads();
#endif
	}

	// stop timing
	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

	// write time and data back to memory
	startClk[tid] = start;
	stopClk[tid] = stop;
	dsink[tid] = ptr1;
#ifdef FINE_GRAINED
	dsink[tid] += tem[tid];
#endif
}

int main(int argc, char **argv)
{
	if (argc > 1) {
		int device = 0;
		device = atoi(argv[1]);
		cudaSetDevice(device);
	}
	uint32_t *startClk = (uint32_t *)malloc(THREADS_NUM * sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t *)malloc(THREADS_NUM * sizeof(uint32_t));
	uint64_t *dsink = (uint64_t *)malloc(THREADS_NUM * sizeof(uint64_t));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	uint64_t *posArray_g;
	uint64_t *fakeArray_g;
	uint64_t *dsink_g;

	gpuErrchk(cudaMalloc(&startClk_g, THREADS_NUM * sizeof(uint32_t)));
	gpuErrchk(cudaMalloc(&stopClk_g, THREADS_NUM * sizeof(uint32_t)));
	gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&fakeArray_g, ARRAY_SIZE * sizeof(uint64_t)));
	gpuErrchk(cudaMalloc(&dsink_g, THREADS_NUM * sizeof(uint64_t)));

	init_data<<<BLOCKS, THREADS_PER_BLOCK>>>(posArray_g);
	//init_data<<<BLOCKS, THREADS_PER_BLOCK>>>(fakeArray_g); // evict cache
	mem_lat<<<1, THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	// mem_lat<<<1, THREADS_NUM>>>(startClk_g, stopClk_g, posArray_g, dsink_g);

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(startClk, startClk_g, THREADS_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(stopClk, stopClk_g, THREADS_NUM * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(dsink, dsink_g, THREADS_NUM * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(dsink, fakeArray_g, THREADS_NUM * sizeof(uint64_t), cudaMemcpyDeviceToHost));

#ifndef FINE_GRAINED
	printf("Mem latency = %12.4f cycles \n", (float)(stopClk[0] - startClk[0]) / (float)(ITERS));
	printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);
#endif

	gpuErrchk(cudaFree(startClk_g));
	gpuErrchk(cudaFree(stopClk_g));
	gpuErrchk(cudaFree(posArray_g));
	gpuErrchk(cudaFree(fakeArray_g));
	gpuErrchk(cudaFree(dsink_g));
	return 0;
}
