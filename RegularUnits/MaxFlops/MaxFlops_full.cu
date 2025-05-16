#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 256
#define SM_NUM 132
#define BLOCKS_NUM 1320*4*2
#define THREADS_PER_SM (THREADS_PER_BLOCK*BLOCKS_NUM/SM_NUM)
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
//#define REPEAT_TIMES 32768000
#define REPEAT_TIMES 16384

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


template <class T>
__global__ void max_flops(T *data1, T *data2, T *res) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	register T s1 = data1[gid];
	register T s2 = data2[gid];
	register T s3 = s1;
	register T s4 = s2;
	register T result = 1;
	// synchronize all threads
	// asm volatile ("bar.sync 0;");

	// start timing
	// uint32_t start = 0;
	// asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

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
	// asm volatile("bar.sync 0;");

	// // stop timing
	// uint32_t stop = 0;
	// asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");

	// write time and data back to memory
	result = s1 + s2 + s3 + s4;
	res[gid] = result;
}

int main(){
	float *data1 = (float*) malloc(TOTAL_THREADS*sizeof(float));
	float *data2 = (float*) malloc(TOTAL_THREADS*sizeof(float));
	float *res = (float*) malloc(TOTAL_THREADS*sizeof(float));

	float *data1_g;
	float *data2_g;
	float *res_g;

	for (uint32_t i=0; i<TOTAL_THREADS; i++) {
		data1[i] = (float)i;
		data2[i] = (float)i;
	}

	gpuErrchk( cudaMalloc(&data1_g, TOTAL_THREADS*sizeof(float)) );
	gpuErrchk( cudaMalloc(&data2_g, TOTAL_THREADS*sizeof(float)) );
	gpuErrchk( cudaMalloc(&res_g, TOTAL_THREADS*sizeof(float)) );

	gpuErrchk( cudaMemcpy(data1_g, data1, TOTAL_THREADS*sizeof(float), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(data2_g, data2, TOTAL_THREADS*sizeof(float), cudaMemcpyHostToDevice) );
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	max_flops<float><<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(data1_g, data2_g, res_g);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(res, res_g, TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float seconds = milliseconds / 1000;
	float Gflops;
	Gflops = (float)(REPEAT_TIMES*(unsigned long)TOTAL_THREADS*8)/seconds/1e9;
	printf("FLOPS = %f G\n", Gflops);

	return 0;
} 

