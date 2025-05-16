#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <mma.h>
#include <stdio.h>
#include <type_traits>
#include <stdlib.h>

#include "../../include/util.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif 

#ifndef ILP
#define ILP 4
#endif 

#ifndef ITERs
#define ITERs 10000000
#endif

#define H2D  cudaMemcpyHostToDevice
#define D2H  cudaMemcpyDeviceToHost

using u32 = uint32_t;
using u64 = uint64_t;
using f16 = half;
using f32 = float;

#define mma_inst_tf32_f32(ii2,ii4) \
{\
    asm volatile(\
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
        : "=f"(D[ii4]), "=f"(D[ii4+1]), "=f"(D[ii4+2]), "=f"(D[ii4+3]) \
        :   \
        "r"(a_frag[ii4]), "r"(a_frag[ii4+1]), "r"(a_frag[ii4+2]), "r"(a_frag[ii4+3]), \
        "r"(b_frag[ii2]), "r"(b_frag[ii2+1]), \
        "f"(c_frag[ii4]), "f"(c_frag[ii4+1]), "f"(c_frag[ii4+2]), "f"(c_frag[ii4+3])\
    ); \
}

#define mma_inst_f16_f32(ii1,ii2,ii4) \
{\
    asm volatile(\
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n" \
        : "=f"(D[ii4]), "=f"(D[ii4+1]), "=f"(D[ii4+2]), "=f"(D[ii4+3]) \
        :   \
        "r"(a_frag[ii2]), "r"(a_frag[ii2+1]), \
        "r"(b_frag[ii1]), \
        "f"(c_frag[ii4]), "f"(c_frag[ii4+1]), "f"(c_frag[ii4+2]), "f"(c_frag[ii4+3])\
    ); \
}

#define mma_inst_f16_f16(ii1,ii2) \
{\
    asm volatile(\
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n" \
        : "=r"(D[ii2]), "=r"(D[ii2+1]) \
        :   \
        "r"(a_frag[ii2]), "r"(a_frag[ii2+1]), \
        "r"(b_frag[ii1]), \
        "r"(c_frag[ii2]), "r"(c_frag[ii2+1])\
    ); \
}

__global__ void  warm_up() {
    printf("warm up\n");
}


template<class T1, class T2>
__global__  void bench(T1* A, T1* B, T2* C, u64 *start_clk, u64 *end_clk) {
    u64 start, end;

    u32 bid = blockIdx.x;
    u32 bdim = blockDim.x;
    u32 tid = threadIdx.x;
    u32 gtid = bid * bdim + tid;
    u32 gwid = gtid / 32;

    T1 a_frag_float[4 * ILP];
    T1 b_frag_float[2 * ILP];
    T2 c_frag_float[4 * ILP];

    for (int ii = 0; ii < 4 * ILP; ii++) {
        a_frag_float[ii] = A[gwid*128+ii];
        c_frag_float[ii] = 0.0f;
    }

    for (int ii = 0; ii < 2 * ILP; ii++) b_frag_float[ii] = B[gwid*64+ii];

    if (std::is_same<T2, f32>::value) {
        if (std::is_same<T1, f32>::value) {
            const u32* a_frag = reinterpret_cast<const u32*>(&a_frag_float[0]);
            const u32* b_frag = reinterpret_cast<const u32*>(&b_frag_float[0]); // tf32
            f32 *c_frag = reinterpret_cast<f32*>(&c_frag_float[0]);
            f32 *D = reinterpret_cast<f32*>(&c_frag_float[0]);

            __syncthreads();
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory"); 

            for (int ii = 0; ii < ITERs; ii++) {
#pragma unroll
                for (int inst = 0; inst < ILP; inst++) {
                    mma_inst_tf32_f32(inst*2, inst*4);
                }
                __syncwarp();
            }
            __syncthreads();
        } else if (std::is_same<T1, f16>::value) {
            const u32* a_frag = reinterpret_cast<const u32*>(&a_frag_float[0]);
            const u32* b_frag = reinterpret_cast<const u32*>(&b_frag_float[0]);
            f32* c_frag = reinterpret_cast<f32*>(&c_frag_float[0]);
            f32* D = reinterpret_cast<f32*>(&c_frag_float[0]);
            __syncthreads();
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory"); 
            for (int ii = 0; ii < ITERs; ii++) {
#pragma unroll
                for (int inst = 0; inst < ILP; inst++) {
                    mma_inst_f16_f32(inst, inst*2, inst*4);
                }
                __syncwarp();
            }
            __syncthreads();
        } else assert(0);
    } else if (std::is_same<T2, f16>::value) {
         if (std::is_same<T1, f16>::value) {
            const u32* a_frag = reinterpret_cast<const u32*>(&a_frag_float[0]);
            const u32* b_frag = reinterpret_cast<const u32*>(&b_frag_float[0]);
            u32* c_frag = reinterpret_cast<u32*>(&c_frag_float[0]);
            u32* D = reinterpret_cast<u32*>(&c_frag_float[0]);
            __syncthreads();
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory"); 
            for (int ii = 0; ii < ITERs; ii++) {
#pragma unroll
                for (int inst = 0; inst < ILP; inst++) {
                    mma_inst_f16_f16(inst, inst*2);
                }
                __syncwarp();
            }
            __syncthreads();

        } else assert(0);
    } else assert(0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    for (int ii = 0; ii < ILP * 4; ii++) {
        C[gwid * 128 + ii] = c_frag_float[ii];
    }

    start_clk[gtid] = start;
    end_clk[gtid] = end;
}

template<class T1, class T2>
void run(u32 threads_per_block) {
    u32 warp_num = threads_per_block / WARP_SIZE;
    dim3 grid_dim = SMs;
    dim3 block_dim = threads_per_block;

    // A 16 * 8; B 8 * 8; C 16 * 8 each warp
    T1 *data_Ad, *data_Bd;
    T2 *data_Cd;
    T1 *data_Ah, *data_Bh;
    T2 *data_Ch;
    u64 *start_clkd, *end_clkd;
    u64 *start_clkh, *end_clkh, *clk;

    u32 sizeA = 16 * 8 * warp_num * sizeof(T1) * SMs;
    u32 sizeB = 8 * 8 * warp_num * sizeof(T1) * SMs;
    u32 sizeC = 16 * 8 * warp_num * sizeof(T2) * SMs;
    u32 sizeclk = threads_per_block * SMs * sizeof(u64);

    data_Ah = reinterpret_cast<T1*>(malloc(sizeA));
    data_Bh = reinterpret_cast<T1*>(malloc(sizeB));
    data_Ch = reinterpret_cast<T2*>(malloc(sizeC));
    start_clkh = reinterpret_cast<u64*>(malloc(sizeclk));
    end_clkh = reinterpret_cast<u64*>(malloc(sizeclk));
    clk = reinterpret_cast<u64*>(malloc(sizeclk));

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Ad), sizeA));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Bd), sizeB));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Cd), sizeC));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&start_clkd), sizeclk));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&end_clkd), sizeclk));

    u32 numA = sizeA / sizeof(T1);
    u32 numB = sizeB / sizeof(T1);

    for (u32 ii = 0; ii < numA; ii++) {
        if (ii < numB) data_Bh[ii] = ii+1;
        data_Ah[ii] = ii + 1;
        data_Ch[ii] = ii + 2;
    }

    gpuErrchk(cudaMemcpy(data_Ad, data_Ah, sizeA, H2D));
    gpuErrchk(cudaMemcpy(data_Bd, data_Bh, sizeB, H2D));
    gpuErrchk(cudaMemcpy(data_Cd, data_Ch, sizeC, H2D));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    bench<<<grid_dim, block_dim>>>(data_Ad, data_Bd, data_Cd, start_clkd, end_clkd);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    f32 time;
    cudaEventElapsedTime(&time, start, end);
    
    std::cout << "Time : " << time << " ms" << std::endl;
    std::cout << "Throughput(whole kernel) : " << f32(16) * 8 * 8 * 2 * warp_num * SMs * ILP / 1e9 * ITERs / time << " TFLOPS/TOPS" << std::endl;

    gpuErrchk(cudaMemcpy(start_clkh, start_clkd, sizeclk, D2H));
    gpuErrchk(cudaMemcpy(end_clkh, end_clkd, sizeclk, D2H));

    for (int ii = 0; ii < sizeclk/sizeof(u64); ii++) {
        clk[ii] = end_clkh[ii] - start_clkh[ii];
    }

    u64 clock_latency = 
              *std::max_element(clk, clk+threads_per_block * SMs);


    std::cout << "Latency = "  << clock_latency << std::endl;
    std::cout << "Throughput(mma inst) = " << f32(16) * 8 * 8 * 2 * warp_num * SMs * ILP / 1e12 * GPUFreq * ITERs / clock_latency << " TFLOPS/TOPS" <<std::endl;

    cudaFree(data_Ad);
    cudaFree(data_Bd);
    cudaFree(data_Cd);
    cudaFree(start_clkd);
    cudaFree(end_clkd);
}

int main() {
    InitGPUDeviceProperty();
    warm_up<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "**********START MMA F16 F16 BENCH***********" << std::endl;
    for (u32 threads_per_block = 32; threads_per_block <= 1024; threads_per_block*=2) {
        // 32 64 128 256 512 1024
        std::cout << "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 TEST with ILP = " << ILP << ", warp num = " << threads_per_block / WARP_SIZE << ", SMs = " << SMs << ", GPUFreq = " << GPUFreq << std::endl;
        run<f16, f16>(threads_per_block);
    }
    std::cout << std::endl;
    std::cout << "**********START MMA F16 F32 BENCH***********" << std::endl;
    for (u32 threads_per_block = 32; threads_per_block <= 1024; threads_per_block*=2) {
        // 32 64 128 256 512 1024
        std::cout << "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 TEST with ILP = " << ILP << ", warp num = " << threads_per_block / WARP_SIZE << ", SMs = " << SMs << ", GPUFreq = " << GPUFreq << std::endl;
        run<f16, f32>(threads_per_block);
    }
    std::cout << std::endl;
    std::cout << "**********START MMA TF32 F32 BENCH***********" << std::endl;
    for (u32 threads_per_block = 32; threads_per_block <= 1024; threads_per_block*=2) {
        // 32 64 128 256 512 1024
        std::cout << "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 TEST with ILP = " << ILP << ", warp num = " << threads_per_block / WARP_SIZE << ", SMs = " << SMs << ", GPUFreq = " << GPUFreq << std::endl;
        run<f32, f32>(threads_per_block);
    }
}