#ifndef __INCLUDE_UTIL_H__
#define __INCLUDE_UTIL_H__

#include <cstring>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

int GPUFreq = -1;
int SMs = -1;

void InitGPUDeviceProperty() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    GPUFreq = prop.clockRate*1000; // scale
    cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, 0);
}

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

void init_indics(char *indics, int size, int bit_per_indics) {
  memset(indics, 0, size);
  if (bit_per_indics == 4) {
    int num0 = 10;
    int num1 = 4;
    // Matrix A with tf32 type
    for (int ii = 0; ii < size; ii++) {
      for (int jj = 0; jj < 8/bit_per_indics; jj++) {
        int seed = std::rand()%2;
        if (seed == 0) 
          indics[ii] += num0 << (4*jj);
        else 
          indics[ii] += num1 << (4*jj);
      }
    }
  } else if (bit_per_indics == 2) {
    // Matrix B with bf16 or u8/i8 type
    for (int ii = 0; ii < size; ii++) {
      for (int jj = 0; jj < 8/bit_per_indics; jj++) {
        int seed = std::rand()%4;
        switch(seed) {
          case 0:
            break;
          case 1:
            indics[ii] += 1 << (2*jj);
            break;
          case 2:
            indics[ii] += 2 << (2*jj);
            break;
          case 3:
            indics[ii] += 3 << (2*jj);
            break;
          default:
            break;
        }
      }
    }
  }
  else assert(0);
}

#endif
