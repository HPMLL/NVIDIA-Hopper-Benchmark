#include <helper_cuda.h>
#include <cooperative_groups.h>


// Distributed Shared memory kernel
__global__ void dsm_sm2sm_lat_kernel(uint *d_Data, uint array_size)
{
    extern __shared__ uint smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    clock_t start, end;
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cluster_block_rank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    if (threadIdx.x == 0) {
        d_Data[cluster_block_rank] = smid;
    }

    uint dst_block_rank = 0;
    uint *dst_smem = 0;


    for (int i = 0; i < array_size; i++)
    {
        smem[i] = i + 1;
    }


    cluster.sync();

    for (uint b = 0; b < cluster_size; ++b) {
        if (b == blockIdx.x) {
            for (uint i = 0; i < cluster_size; ++i) {
                uint index = 0;
                dst_block_rank = i;
                dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
                start = clock();
                for (int j = 0; j < array_size; j++)
                {
                    index = dst_smem[index];
                }
                end = clock();

                // aviod compile optimization

                d_Data[array_size - 2] += index; 

                if ((threadIdx.x == 0)) {
                    d_Data[cluster_size + cluster_size * cluster_block_rank + dst_block_rank] = (end - start) / array_size;
                }
                // printf("Rank %2u in SM %3u -> rank %2u in SM %3u: %ld\n", clusterBlockRank, smid, dst_block_rank, index, (end - start)/array_size );       
            }
        }        
        cluster.sync();

    }

    // aviod compile optimization
    atomicAdd(d_Data + array_size - 1, smem[tid + cluster_block_rank + dst_block_rank]);



}

extern "C" void dsm_sm2sm_latency(void *d_Data, uint arraySize, uint cluster_size)
{
    uint threads_per_block = 1;

    cudaLaunchConfig_t config = {0};

    config.gridDim = cluster_size;
    config.blockDim = threads_per_block;

    config.dynamicSmemBytes = arraySize * sizeof(uint);

    // CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
    cudaFuncSetAttribute((void *)dsm_sm2sm_lat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    cudaFuncSetAttribute((void *)dsm_sm2sm_lat_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute((void *)dsm_sm2sm_lat_kernel, cudaFuncAttributeClusterSchedulingPolicyPreference, 0);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, dsm_sm2sm_lat_kernel, (uint *)d_Data, arraySize);
    getLastCudaError("dsm_sm2sm_lat_kernel() execution failed\n");
}
