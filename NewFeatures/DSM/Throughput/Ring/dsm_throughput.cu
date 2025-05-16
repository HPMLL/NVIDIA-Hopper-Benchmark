#include <helper_cuda.h>
#include <cooperative_groups.h>
#include "../../dsm_common.h"


__global__ void dsm_sm2sm_thrpt_kernel(uint *d_Data, uint repeat_times, uint array_size_bytes, uint stride)
{
    extern __shared__ uint smem[];
    namespace cg = cooperative_groups;
    int tid = threadIdx.x;
    auto block_size = blockDim.x;

    // uint32_t smid, cluster_id;
    // asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    // asm volatile("mov.u32 %0, %clusterid.x;" : "=r"(cluster_id));
    // // uint32_t ctaid;
    // // asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(ctaid));
    // if (tid == 0) {
    //     printf("%d, %d, %d\n", blockIdx.x, smid, cluster_id);
    // }


    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    auto array_size = array_size_bytes / sizeof(uint);

    for (int i = tid; i < array_size; i += block_size) {
        smem[i] = (i + stride) % array_size;
    }

    cluster.sync();

    uint dst_block_rank = 0;
    uint *dst_smem = 0;

    dst_block_rank = (clusterBlockRank + 1) % cluster_size;
    // dst_block_rank = (clusterBlockRank / 2) * 2 + (clusterBlockRank + 1) % 2;
    dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    register uint temp = tid;

    for (uint32_t i = 0; i < repeat_times; i++) {
        temp = dst_smem[temp];
    }


    cluster.sync();

    d_Data[tid] = temp;
}


void dsm_sm2sm_thrpt(void *d_Data, uint arraySize, uint cluster_size, uint block_num, uint block_size, uint repeat_times)
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = block_num;
    config.blockDim = block_size;

    // dynamic shared memory size is per block.
    config.dynamicSmemBytes = arraySize;

    // CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeClusterSchedulingPolicyPreference, 0);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    // int number_clusters, potential_cluster_size;
    // checkCudaErrors(cudaOccupancyMaxActiveClusters(&number_clusters, dsm_sm2sm_thrpt_kernel, &config));
    // printf("number_clusters: %d\n", number_clusters);
    // checkCudaErrors(cudaOccupancyMaxPotentialClusterSize(&potential_cluster_size, dsm_sm2sm_thrpt_kernel, &config));
    // printf("potential_cluster_size: %d\n", potential_cluster_size);


    cudaLaunchKernelEx(&config, dsm_sm2sm_thrpt_kernel, (uint *)d_Data, repeat_times, arraySize, 1024);
    getLastCudaError("dsm_sm2sm_thrpt_kernel() execution failed\n");
}

