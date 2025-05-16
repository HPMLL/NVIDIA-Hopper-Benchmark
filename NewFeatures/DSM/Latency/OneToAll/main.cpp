/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h> // helper for shared that are common to CUDA Samples

// project include
#include "../../dsm_common.h"

int main(int argc, char **argv)
{
    uint *d_Data;
    uint arraySize = 1024;

    uint *h_Data = new uint[arraySize];

    checkCudaErrors(cudaMalloc((void **)&d_Data, arraySize * sizeof(uint)));

    printf("Measure latency of SM to SM for %ld bytes...\n\n",
           arraySize * sizeof(uint));

    printf("Benchmarking time...\n");

    for (auto cluster_size = 2; cluster_size <= 16; ++ cluster_size) {
        printf("\nCluster size: %d\n", cluster_size);

        dsm_sm2sm_latency(d_Data, arraySize, cluster_size);

        checkCudaErrors(cudaMemcpy(h_Data, d_Data, arraySize * sizeof(uint), cudaMemcpyDeviceToHost));

        // i -> j
        // [smid, smid, smid, smid, 0-0 latency, 0-1 latency, .....]
        for (uint i = 0; i < cluster_size; ++i) {
            for (uint j = 0; j < cluster_size; ++j) {
                auto latency =  h_Data[cluster_size + i * cluster_size + j];
                printf("Rank %2u in SM %3u -> rank %2u in SM %3u: %u\n", i, h_Data[i], j, h_Data[j], latency);       
            }
        }
    }


    checkCudaErrors(cudaFree(d_Data));
    delete h_Data;
}
