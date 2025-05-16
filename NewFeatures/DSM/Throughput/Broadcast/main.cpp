// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h> // helper for shared that are common to CUDA Samples

// project include
#include "../../dsm_common.h"

uint arraySize = 1024 * 16; //bytes
constexpr uint cluster_size_list[] = {2, 4, 8, 16};
uint block_num = 114 * 220 * 2;
uint block_size = 1024;
uint repeat_times = 4096 * 16;


// template <std::size_t I, size_t N>
// typename std::enable_if<I == N, void>::type
// run_dsm_sm2sm_thrpt_impl(void* d_Data, uint arraySize, uint cluster_size, uint block_num, uint block_size, uint repeat_times, StopWatchInterface **hTimer) {
// }

// template <std::size_t I = 0, size_t N = sizeof(ilp_list)/sizeof(ilp_list[0])>
// typename std::enable_if<I < N, void>::type
// run_dsm_sm2sm_thrpt_impl(void* d_Data, uint arraySize, uint cluster_size, uint block_num, uint block_size, uint repeat_times, StopWatchInterface **hTimer) {
//     checkCudaErrors(cudaDeviceSynchronize());
//     sdkResetTimer(hTimer);
//     sdkStartTimer(hTimer);
//     dsm_sm2sm_thrpt<ilp_list[I]>(d_Data, arraySize, cluster_size, block_num, block_size, repeat_times);
//     cudaDeviceSynchronize();
//     sdkStopTimer(hTimer);
//     double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(hTimer) / (double)repeat_times;
//     printf("dsm_sm2sm_thrpt() with ILP %d time (average) : %.5f sec\n", ilp_list[I], dAvgSecs);
//     run_dsm_sm2sm_thrpt_impl<I+1, N>(d_Data, arraySize, cluster_size, block_num, block_size, repeat_times, hTimer);
// }

// template <size_t N>
// void run_dsm_sm2sm_thrpt_impl(void* d_Data, uint arraySize, uint cluster_size, uint block_num, uint block_size, uint repeat_times, StopWatchInterface **hTimer, const uint (&ilp_list)[N]) {
//     for (size_t i = 0; i < N; ++i) {
//         checkCudaErrors(cudaDeviceSynchronize());
//         sdkResetTimer(hTimer);
//         sdkStartTimer(hTimer);
//         if constexpr (ilp_list[i] > 0) {
//             dsm_sm2sm_thrpt<ilp_list[i]>(d_Data, arraySize, cluster_size, block_num, block_size, repeat_times);
//         }
//         cudaDeviceSynchronize();
//         sdkStopTimer(hTimer);
//         double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(hTimer) / (double)repeat_times;
//         printf("dsm_sm2sm_thrpt() with ILP %d time (average) : %.5f sec\n", ilp_list[i], dAvgSecs);
//     }

// }


int main(int argc, char **argv)
{
    uint *d_Data;

    StopWatchInterface *hTimer = NULL;


    sdkCreateTimer(&hTimer);

    checkCudaErrors(cudaMalloc((void **)&d_Data, arraySize * sizeof(uint)));

    printf("Measure throughput of SM to SM for %ld bytes...\n\n",
           arraySize * sizeof(uint));

    printf("Benchmarking time...\n");


    
    for (auto cluster_size : cluster_size_list) {
        // run_dsm_sm2sm_thrpt_impl(d_Data, cluster_size, &hTimer, std::make_index_sequence<sizeof(ilp_list)/sizeof(ilp_list[0])>{});
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
        dsm_sm2sm_thrpt(d_Data, arraySize, cluster_size, block_num, block_size, repeat_times);
        cudaDeviceSynchronize();
        sdkStopTimer(&hTimer);
        double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);

        uint cluster_num = block_num / cluster_size;
        printf("Cluster size: %u, Throughpht: %.4fTB/s\n", cluster_size, static_cast<float>(repeat_times) * sizeof(uint) * block_size * cluster_num * (cluster_size - 1) / 1000000000000.0 / dAvgSecs);
    }

    

    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_Data));
}
