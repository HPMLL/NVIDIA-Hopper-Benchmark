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

#define StrideA 16*32
#define StrideB 8*32
#define StrideC 16*8

using u32 = uint32_t;
using u64 = uint64_t;
using f16 = half;
using f32 = float;
using i8 = int8_t;
using i32 = int32_t;

#define mma_inst_i4_i32(ii1,ii2,ii4) \
{\
    asm volatile(\
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n" \
        : "=r"(D[ii4]), "=r"(D[ii4+1]), "=r"(D[ii4+2]), "=r"(D[ii4+3]) \
        :   \
        "r"(a_frag[ii2]), "r"(a_frag[ii2+1]), \
        "r"(b_frag[ii1]), \
        "r"(c_frag[ii4]), "r"(c_frag[ii4+1]), "r"(c_frag[ii4+2]), "r"(c_frag[ii4+3])\
    ); \
}

#define mma_inst_i8_i32(ii2,ii4) \
{\
    asm volatile(\
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
        : "=r"(D[ii4]), "=r"(D[ii4+1]), "=r"(D[ii4+2]), "=r"(D[ii4+3]) \
        :   \
        "r"(a_frag[ii4]), "r"(a_frag[ii4+1]),"r"(a_frag[ii4+2]), "r"(a_frag[ii4+3]),  \
        "r"(b_frag[ii2]), "r"(b_frag[ii2+1]), \
        "r"(c_frag[ii4]), "r"(c_frag[ii4+1]), "r"(c_frag[ii4+2]), "r"(c_frag[ii4+3])\
    ); \
}

__global__ void  warm_up() {
    printf("warm up\n");
}


__global__  void bench(i8* A, i8* B, i32* C, u64* clk_s, u64* clk_e, int operand) {
    u64 start, end;

    u32 bid = blockIdx.x;
    u32 bdim = blockDim.x;
    u32 tid = threadIdx.x;
    u32 gtid = bid * bdim + tid;
    u32 gwid = gtid / 32;

    i8 a_frag_float[16 * ILP];
    i8 b_frag_float[8 * ILP];
    i32 c_frag_float[4 * ILP];

    for (int ii = 0; ii < 16 * ILP; ii++) {
        a_frag_float[ii] = A[gwid*StrideA+ii];
    }
    for (int ii = 0; ii < 8 * ILP; ii++) {
        b_frag_float[ii] = B[gwid*StrideB+ii];
    }
    for (int ii = 0; ii < 4 * ILP; ii++) {
        c_frag_float[ii] = 0.0f;
    }

    if (operand == 4) {
        const u32* a_frag = reinterpret_cast<const u32*>(&a_frag_float[0]);
        const u32* b_frag = reinterpret_cast<const u32*>(&b_frag_float[0]);
        i32* c_frag = reinterpret_cast<i32*>(&c_frag_float[0]);
        i32* D = reinterpret_cast<i32*>(&c_frag_float[0]);
        __syncthreads();
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory"); 
        for (int ii = 0; ii < ITERs; ii++) {
#pragma unroll
            for (int inst = 0; inst < ILP; inst++) {
                mma_inst_i4_i32(inst, inst*2, inst*4);
            }
            __syncwarp();
        }
        __syncthreads();
    } else if (operand == 8) {
        const u32* a_frag = reinterpret_cast<const u32*>(&a_frag_float[0]);
        const u32* b_frag = reinterpret_cast<const u32*>(&b_frag_float[0]);
        i32* c_frag = reinterpret_cast<i32*>(&c_frag_float[0]);
        i32* D = reinterpret_cast<i32*>(&c_frag_float[0]);
        __syncthreads();
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory"); 
        for (int ii = 0; ii < ITERs; ii++) {
#pragma unroll
            for (int inst = 0; inst < ILP; inst++) {
                mma_inst_i8_i32(inst*2, inst*4);
            }
            __syncwarp();
        }
        __syncthreads();
    } else assert(0);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    for (int ii = 0; ii < ILP * 4; ii++) {
        C[gwid * StrideC + ii] = c_frag_float[ii];
    }

    clk_s[gtid] = start;
    clk_e[gtid] = end;
}

void run(u32 threads_per_block, int operand) {
    u32 warp_num = threads_per_block / WARP_SIZE;
    dim3 grid_dim = SMs;
    dim3 block_dim = threads_per_block;

    // A 16 * 16; B 16 * 8; C 16 * 8 each warp
    i8 *data_Ad, *data_Bd;
    i32 *data_Cd;
    i8 *data_Ah, *data_Bh;
    i32 *data_Ch;
    u64 *start_clkd, *end_clkd;
    u64 *start_clkh, *end_clkh, *clk;

    u32 sizeA = StrideA * warp_num * sizeof(i8) * SMs;
    u32 sizeB = StrideB * warp_num * sizeof(i8) * SMs;
    u32 sizeC = StrideC * warp_num * sizeof(i32) * SMs;
    u32 sizeclk = threads_per_block * SMs * sizeof(u64);

    data_Ah = reinterpret_cast<i8*>(malloc(sizeA));
    data_Bh = reinterpret_cast<i8*>(malloc(sizeB));
    data_Ch = reinterpret_cast<i32*>(malloc(sizeC));
    start_clkh = reinterpret_cast<u64*>(malloc(sizeclk));
    end_clkh = reinterpret_cast<u64*>(malloc(sizeclk));
    clk = reinterpret_cast<u64*>(malloc(sizeclk));

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Ad), sizeA));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Bd), sizeB));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Cd), sizeC));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&start_clkd), sizeclk));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&end_clkd), sizeclk));

    for (u32 ii = 0; ii < StrideA; ii++) {
        data_Ah[ii] = ii + 1;
    }
    for (u32 ii = 0; ii < StrideB; ii++) {
        data_Bh[ii] = ii + 1;
    }
    for (u32 ii = 0; ii < StrideC; ii++) {
        data_Ch[ii] = ii + 2;
    }

    gpuErrchk(cudaMemcpy(data_Ad, data_Ah, sizeA, H2D));
    gpuErrchk(cudaMemcpy(data_Bd, data_Bh, sizeB, H2D));
    gpuErrchk(cudaMemcpy(data_Cd, data_Ch, sizeC, H2D));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    bench<<<grid_dim, block_dim>>>(data_Ad, data_Bd, data_Cd, start_clkd, end_clkd, operand);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    f32 time;
    cudaEventElapsedTime(&time, start, end);
    
    std::cout << "Time : " << time << " ms" << std::endl;
    std::cout << "Throughput(whole kernel) : " << f32(16) * 8 * 32 * 2 * warp_num * SMs * ILP / 1e9 * ITERs / time << " TFLOPS/TOPS" << std::endl;

    gpuErrchk(cudaMemcpy(start_clkh, start_clkd, sizeclk, D2H));
    gpuErrchk(cudaMemcpy(end_clkh, end_clkd, sizeclk, D2H));

    for (int ii = 0; ii < sizeclk/sizeof(u64); ii++) {
        clk[ii] = end_clkh[ii] - start_clkh[ii];
    }

    u64 clock_latency = 
              *std::max_element(clk, clk+threads_per_block * SMs);


    std::cout << "Latency = "  << clock_latency << std::endl;
    std::cout << "Throughput(mma inst) = " << f32(16) * 8 * 32 * 2 * warp_num * SMs * ILP / 1e12 * GPUFreq * ITERs / clock_latency << " TFLOPS/TOPS" <<std::endl;

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

    std::cout << "**********START MMA INT4 INT32 BENCH***********" << std::endl;
    for (u32 threads_per_block = 32; threads_per_block <= 1024; threads_per_block*=2) {
        // 32 64 128 256 512 1024
        std::cout << "mma.sync.aligned.m16n8k32.row.col.i32.i4.i4.i32 TEST with ILP = " << ILP << ", warp num = " << threads_per_block / WARP_SIZE << ", SMs = " << SMs << ", GPUFreq = " << GPUFreq << std::endl;
        run(threads_per_block, 4);
    }
    std::cout << std::endl;

    std::cout << "**********START MMA INT8 INT32 BENCH***********" << std::endl;
    for (u32 threads_per_block = 32; threads_per_block <= 1024; threads_per_block*=2) {
        // 32 64 128 256 512 1024
        std::cout << "mma.sync.aligned.m16n8k32.row.col.i32.i8.i8.i32 TEST with ILP = " << ILP << ", warp num = " << threads_per_block / WARP_SIZE << ", SMs = " << SMs << ", GPUFreq = " << GPUFreq << std::endl;
        run(threads_per_block, 8);
    }
}
