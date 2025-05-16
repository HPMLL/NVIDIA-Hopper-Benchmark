#include <cuda/barrier>
#include "../../util.h"


using barrier = cuda::barrier<cuda::thread_scope_block>;


typedef float dtype;

#define ARRAY_SIZE (4 * 1024*1024*(1024/sizeof(dtype))) // GB
constexpr int BLOCKS[] = {114, 228, 342, 456};
#define THREADS_PER_BLOCK 1024

constexpr int LOAD_SIZE_LIST[] = {1*1024, 2*1024, 4*1024, 8*1024, 12*1024, 16*1024}; //bytes
constexpr int LOAD_SIZE = LOAD_SIZE_LIST[2];

__global__ void init_data(dtype * array) {
    uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
    auto total_threads = blockDim.x * gridDim.x;

	for (uint32_t i = uid; i < ARRAY_SIZE; i += total_threads) {
		array[i] = uid;
    }
}

__global__ void tma_bw(dtype * volatile array, dtype *dsink)
{

    uint32_t tid = threadIdx.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;
    // dtype temp_res = 0;

    __shared__ alignas(16) dtype smem[LOAD_SIZE/sizeof(dtype)];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (tid == 0) {
        init(&bar, blockDim.x);                    // a)
        asm volatile("fence.proxy.async.shared::cta;");     // b)
        
        for (int i = uid * (LOAD_SIZE / sizeof(dtype)); i < ARRAY_SIZE; i += gridDim.x * blockDim.x * (LOAD_SIZE / sizeof(dtype))) {

            auto ptr = array + i;

            asm volatile(
                "{\t\n"
                //"discard.L2 [%1], 128;\n\t"
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[%0], [%1], %2, [%3]; // 1a. unicast\n\t"
                "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
                "}"
                :
                //: "r"(static_cast<unsigned>(__cvta_generic_to_shared(ptr))), "l"(ptr[0]), "n"(cuda::aligned_size_t<16>(LOAD_SIZE)), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem))), "l"(ptr), "n"(LOAD_SIZE), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                : "memory"); 


            // 3b. All threads arrive on the barrier
            barrier::arrival_token token = bar.arrive();

            // 3c. Wait for the data to have arrived.
            bar.wait(std::move(token));
            //temp_res += smem[0];
        }


    }


}

int main() {

    for (int i = 0; i < sizeof(BLOCKS)/sizeof(int); ++i) {
        printf("Block size = %d, Load size = %d KB\n", BLOCKS[i], LOAD_SIZE/1024);
        dtype *dsink = (dtype *)malloc(sizeof(dtype));

        dtype *array_g;
        dtype *dsink_g;

        CUDA_CHECK(cudaMalloc(&array_g, sizeof(dtype) * ARRAY_SIZE));
        CUDA_CHECK(cudaMalloc(&dsink_g, sizeof(dtype)));

        init_data<<<BLOCKS[i], THREADS_PER_BLOCK>>>(array_g);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        tma_bw<<<BLOCKS[i], 1>>>(array_g, dsink_g);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        CUDA_CHECK(cudaMemcpy(dsink, dsink_g, sizeof(dtype), cudaMemcpyDeviceToHost));
        printf("Total time = %f ms, transfer size = %lu bytes\n", milliseconds, ARRAY_SIZE * sizeof(dtype));
        printf("Throughput: %f GB/s\n", ARRAY_SIZE * sizeof(dtype) / (milliseconds / 1000) / 1024 / 1024 / 1024);
    }

    
}