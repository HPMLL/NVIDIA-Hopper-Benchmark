#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 1
#define THREADS_PER_SM 1
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 4096

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


// template <class T>
// __global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1, T *data2, T *res) {
// 	int gid = blockIdx.x*blockDim.x + threadIdx.x;
// 	register T s1 = data1[gid];
// 	register T s2 = data2[gid];
// 	register T result = 0;

// 	// synchronize all threads
// 	asm volatile ("bar.sync 0;");

// 	// start timing
// 	uint32_t start = 0;
// 	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

// 	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
// 		asm volatile ("{\t\n"
// 				"fma.rn.f32 %0, %1, %2 , %0;\n\t"
// 				"fma.rn.f32 %0, %1, %2 , %0;\n\t"
// 				"fma.rn.f32 %0, %1, %2 , %0;\n\t"
// 				"fma.rn.f32 %0, %1, %2 , %0;\n\t"
// 				"}" : "+f"(result),"+f"(s1),"+f"(s2)
// 		);

// 	}
// 	// synchronize all threads
// 	asm volatile("bar.sync 0;");

// 	// stop timing
// 	uint32_t stop = 0;
// 	asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

// 	// write time and data back to memory
// 	startClk[gid] = start;

// 	stopClk[gid] = stop;
// 	res[gid] = result;
// }

template <class T>
__global__ void max_flops(uint32_t *startClk, uint32_t *stopClk, T *data1, T *data2, T *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register T s1 = data1[gid];
	register T s2 = data2[gid];
	register T s3 = s1;
	register T s4 = s2;
	register T result = 1;

	// synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

	for (int j=0 ; j<REPEAT_TIMES ; ++j) {
		// asm volatile ("{\t\n"
		// 		"fma.rn.f32 %0, %1, %2 , %0;\n\t"
		// 		"fma.rn.f32 %0, %1, %2 , %0;\n\t"
		// 		"fma.rn.f32 %0, %1, %2 , %0;\n\t"
		// 		"fma.rn.f32 %0, %1, %2 , %0;\n\t"
		// 		"}" : "+f"(result),"+f"(s1),"+f"(s2)
		// );

		s1 += s1 * s2;
		s2 += s2 * s3;
		s3 += s3 * s4;
		s4 += s4 * s1;

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

	max_flops<half><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, data1_g, data2_g, res_g);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(half), cudaMemcpyDeviceToHost) );


	float latency;
	latency = ((float)(stopClk[0]-startClk[0]))/((float)(REPEAT_TIMES*4));
	printf("fp16 latency = %f (clk)\n", latency);
	printf("Total Clk number = %u \n", stopClk[0]-startClk[0]);

	return 0;
} 

