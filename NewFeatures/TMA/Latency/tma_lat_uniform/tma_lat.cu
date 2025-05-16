#include <cuda/barrier>
//#include <cuda/ptx>
#include "../../util.h"


using barrier = cuda::barrier<cuda::thread_scope_block>;
//namespace ptx = cuda::ptx;

#define STRIDE 2 // 16bytes = 2 * uint64_t
#define ITERS 65536			// 1MB of pointer chasing, ITERS*THREADS_NUM*8 bytes
#define ARRAY_SIZE 62914560*1 //(10485760 4090) (52428800 A100) (62914560 H800)  //pointer-chasing array size in 64-bit. total array size is 7 MB which larger than L2 cache size (6 MB in Volta) to avoid l2 cache resident from the copy engine
#define BLOCKS 1			// must be one
#define THREADS_PER_BLOCK 1024
#define TOTAL_THREADS BLOCKS *THREADS_PER_BLOCK
#define LOAD_SIZE 16 //bytes

__global__ void tma_lat(uint32_t *startClk, uint32_t *stopClk, uint64_t * volatile posArray, uint64_t *dsink)
{

    uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
    // initialize pointer-chasing array

	for (uint32_t i = uid; i < (ARRAY_SIZE - STRIDE); i += TOTAL_THREADS)
		posArray[i] = reinterpret_cast<uint64_t>(posArray + i + STRIDE);

	if (uid < STRIDE) {
		// initialize the tail to reference to the head of the array
		posArray[ARRAY_SIZE - (STRIDE - tid)] = reinterpret_cast<uint64_t>(posArray + uid);
    }

    __syncthreads();

    __shared__ alignas(16) uint64_t ptr[LOAD_SIZE/sizeof(uint64_t)];
    
    uint32_t start = 0;
    uint32_t stop = 0;

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (uid == 0)
    {
        init(&bar, 1);                    // a)
        asm volatile("fence.proxy.async.shared::cta;");     // b)
        
        ptr[0] = reinterpret_cast<uint64_t>(posArray);

		// start timing
		asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");
        #pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            asm volatile(
                "{\t\n"
                //"discard.L2 [%1], 128;\n\t"
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[%0], [%1], %2, [%3]; // 1a. unicast\n\t"
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
                "}"
                :
                //: "r"(static_cast<unsigned>(__cvta_generic_to_shared(ptr))), "l"(ptr[0]), "n"(cuda::aligned_size_t<16>(LOAD_SIZE)), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(ptr))), "l"(ptr[0]), "n"(LOAD_SIZE), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "memory"); 


            // 3b. All threads arrive on the barrier
            barrier::arrival_token token = bar.arrive();

            // 3c. Wait for the data to have arrived.
            bar.wait(std::move(token));
            //asm volatile("fence.proxy.async.shared::cta;");
        }

        // stop timing
        asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
    }

    if (uid == 0) {
        // write time and data back to memory
        startClk[tid] = start;
        stopClk[tid] = stop;
        dsink[tid] = ptr[0];
    }

}

int main() {

	uint32_t *startClk = (uint32_t *)malloc(sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t *)malloc(sizeof(uint32_t));
	uint64_t *dsink = (uint64_t *)malloc(sizeof(uint64_t));

	uint32_t *startClk_g;
	uint32_t *stopClk_g;
	uint64_t *posArray_g;
	uint64_t *dsink_g;

	CUDA_CHECK(cudaMalloc(&startClk_g, sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc(&stopClk_g, sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc(&posArray_g, sizeof(uint64_t) * ARRAY_SIZE));
	CUDA_CHECK(cudaMalloc(&dsink_g, sizeof(uint64_t)));

	tma_lat<<<BLOCKS, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, posArray_g, dsink_g);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(startClk, startClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(stopClk, stopClk_g, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(dsink, dsink_g, sizeof(uint64_t), cudaMemcpyDeviceToHost));
	printf("Mem latency = %12.4f cycles \n", (float)(stopClk[0] - startClk[0]) / (float)(ITERS));
	printf("Total Clk number = %u \n", stopClk[0] - startClk[0]);
}