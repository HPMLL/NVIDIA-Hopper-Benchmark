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

#define StrideA 16*4
#define StrideB 8*8
#define StrideC 16*8

using u32 = uint32_t;
using u64 = uint64_t;
using f16 = half;
using f32 = float;
using i8 = int8_t;
using i32 = int32_t;

#define mmasp_inst_tf32_f32(ii2,ii4) \
{\
    asm volatile(\
        "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11},%12,0x0;\n" \
        : "=f"(D[ii4]), "=f"(D[ii4+1]), "=f"(D[ii4+2]), "=f"(D[ii4+3]) \
        :   \
        "r"(a_frag[ii2]), "r"(a_frag[ii2+1]), \
        "r"(b_frag[ii2]), "r"(b_frag[ii2+1]), \
        "f"(c_frag[ii4]), "f"(c_frag[ii4+1]), "f"(c_frag[ii4+2]), "f"(c_frag[ii4+3]), \
        "r"(e_frag) \
    ); \
}

__global__ void  warm_up() {
    printf("warm up\n");
}


template<class T1, class T2>
__global__  void bench(T1* A, T1* B, T2* C, char *indics, u64 *start_clk, u64 *end_clk) {
    u64 start, end;

    u32 bid = blockIdx.x;
    u32 bdim = blockDim.x;
    u32 tid = threadIdx.x;
    u32 gtid = bid * bdim + tid;
    u32 gwid = gtid / 32;

    T1 a_frag_float[2 * ILP];
    T1 b_frag_float[2 * ILP];
    T2 c_frag_float[4 * ILP];
    u32 e_frag = reinterpret_cast<const u32*>(&indics[0])[gwid];

    for (int ii = 0; ii < 2 * ILP; ii++) {
        a_frag_float[ii] = A[gwid*StrideA+ii];
    }
    for (int ii = 0; ii < 2 * ILP; ii++) {
        b_frag_float[ii] = B[gwid*StrideB+ii];
    }
    for (int ii = 0; ii < 4 * ILP; ii++) {
        c_frag_float[ii] = 0.0f;
    }

    if (std::is_same<T2, f32>::value && std::is_same<T1, f32>::value) {
        const u32* a_frag = reinterpret_cast<const u32*>(&a_frag_float[0]);
        const u32* b_frag = reinterpret_cast<const u32*>(&b_frag_float[0]);
        f32* c_frag = reinterpret_cast<f32*>(&c_frag_float[0]);
        f32* D = reinterpret_cast<f32*>(&c_frag_float[0]);
        __syncthreads();
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory"); 
        for (int ii = 0; ii < ITERs; ii++) {
#pragma unroll
            for (int inst = 0; inst < ILP; inst++) {
                mmasp_inst_tf32_f32(inst*2, inst*4);
            }
            __syncwarp();
        }
        __syncthreads();
    } else assert(0);

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end)::"memory");

    for (int ii = 0; ii < ILP * 4; ii++) {
        C[gwid * StrideC + ii] = c_frag_float[ii];
    }

    start_clk[gtid] = start;
    end_clk[gtid] = end;
}

template<class T1, class T2>
void run(u32 threads_per_block) {
    u32 warp_num = threads_per_block / WARP_SIZE;
    dim3 grid_dim = SMs;
    dim3 block_dim = threads_per_block;

    // A 16 * 16; B 16 * 8; C 16 * 8 each warp
    T1 *data_Ad, *data_Bd;
    T2 *data_Cd;
    T1 *data_Ah, *data_Bh;
    T2 *data_Ch;
    u64 *start_clkd, *end_clkd;
    u64 *start_clkh, *end_clkh, *clk;
    char *indics_host, *indics_device;

    u32 sizeA = StrideA * warp_num * sizeof(T1) * SMs;
    u32 sizeB = StrideB * warp_num * sizeof(T1) * SMs;
    u32 sizeC = StrideC * warp_num * sizeof(T2) * SMs;
    u32 sizeclk = threads_per_block * SMs * sizeof(u64);
    u32 sizee = StrideA * warp_num * sizeof(char) * SMs; //surplus

    data_Ah = reinterpret_cast<T1*>(malloc(sizeA));
    data_Bh = reinterpret_cast<T1*>(malloc(sizeB));
    data_Ch = reinterpret_cast<T2*>(malloc(sizeC));
    start_clkh = reinterpret_cast<u64*>(malloc(sizeclk));
    end_clkh = reinterpret_cast<u64*>(malloc(sizeclk));
    clk = reinterpret_cast<u64*>(malloc(sizeclk));
    indics_host = reinterpret_cast<char *>(malloc(sizee));

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Ad), sizeA));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Bd), sizeB));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&data_Cd), sizeC));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&start_clkd), sizeclk));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&end_clkd), sizeclk));
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&indics_device), sizee));

    for (u32 ii = 0; ii < StrideA; ii++) {
        data_Ah[ii] = ii + 1;
    }
    for (u32 ii = 0; ii < StrideB; ii++) {
        data_Bh[ii] = ii + 1;
    }
    for (u32 ii = 0; ii < StrideC; ii++) {
        data_Ch[ii] = ii + 2;
    }

    if (std::is_same<T1,f32>::value) {
        init_indics(indics_host, sizee, 4);
    } else init_indics(indics_host, sizee, 2);

    gpuErrchk(cudaMemcpy(data_Ad, data_Ah, sizeA, H2D));
    gpuErrchk(cudaMemcpy(data_Bd, data_Bh, sizeB, H2D));
    gpuErrchk(cudaMemcpy(data_Cd, data_Ch, sizeC, H2D));
    gpuErrchk(cudaMemcpy(indics_device, indics_host, sizee, H2D));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    bench<<<grid_dim, block_dim>>>(data_Ad, data_Bd, data_Cd, indics_device, start_clkd, end_clkd);

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
    cudaFree(indics_device);
}

int main() {
    InitGPUDeviceProperty();
    warm_up<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "**********START MMASP TF32 F32 BENCH***********" << std::endl;
    for (u32 threads_per_block = 32; threads_per_block <= 1024; threads_per_block*=2) {
        // 32 64 128 256 512 1024
        std::cout << "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 TEST with ILP = " << ILP << ", warp num = " << threads_per_block / WARP_SIZE << ", SMs = " << SMs << ", GPUFreq = " << GPUFreq << std::endl;
        run<f32, f32>(threads_per_block);
    }
}