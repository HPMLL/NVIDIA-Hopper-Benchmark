#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include "mma_sm90_gmma.hpp"
using namespace cute;

constexpr int M = 64;
constexpr int K = 32;

template <int N>
struct GMMA_Selector_SS;

template <>
struct GMMA_Selector_SS<256> {
    using type = SM90_SP_64x256x32_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_SS<128> {
    using type = SM90_SP_64x128x32_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_SS<64> {
    using type = SM90_SP_64x64x32_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_SS<32> {
    using type = SM90_SP_64x32x32_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_SS<16> {
    using type = SM90_SP_64x16x32_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_SS<8> {
    using type = SM90_SP_64x8x32_F32F16F16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <int N>
struct GMMA_Selector_RS;

template <>
struct GMMA_Selector_RS<256> {
    using type = SM90_SP_64x256x32_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_RS<128> {
    using type = SM90_SP_64x128x32_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_RS<64> {
    using type = SM90_SP_64x64x32_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_RS<32> {
    using type = SM90_SP_64x32x32_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_RS<16> {
    using type = SM90_SP_64x16x32_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};
template <>
struct GMMA_Selector_RS<8> {
    using type = SM90_SP_64x8x32_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
};


template <int N>
__global__ void
wgmma_m64nNk16_fmix_SS(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c, uint8_t sm_layout, int exe_time,
                                      uint64_t *startClk, uint64_t *stopClk) {                        
  constexpr int RegCount = 64 * N / 128;
  extern __shared__ char shem[];                                                  
  using GMMA_t = typename GMMA_Selector_SS<N>::type;  
  GMMA_t gmma_instance;
  typename GMMA_t::CRegisters reg_d;                                                         
  typename GMMA_t::ARegisters reg_a;                                                             
                                                                                    
  __half2 *shem_a = (__half2 *)shem;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    shem_a[i] = gm_a[i];                                                               
  }                                                                                  
                                                                                     
  __half2 *shem_b = (__half2 *)(shem + sizeof(__half2) * M * K / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    shem_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   
                                                                                     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shem_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shem_b));        
                                                                                     
  GmmaDescriptor desc_a, desc_b;                                                     
  desc_a.layout_type_ = sm_layout;                                                   
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_a.start_address_ = sm_a_addr >> 4;                                            
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_a.base_offset_ = 0;                                                           
  desc_b.base_offset_ = 0;                                                           
                                                                                     
  desc_a.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
                                                                                     
  desc_a.stride_byte_offset_ = (2 * 8 * 16 * sizeof(__half)) >> 4;                    
  desc_b.stride_byte_offset_ = (4 * 8 * 8 * sizeof(__half)) >> 4;                
                                                                                     
  reg_a[0] = desc_a.desc_;   
  uint32_t metaE = 0x44444444;     
  uint64_t start = 0;
  uint64_t stop = 0;
  asm volatile("bar.sync 0;");  
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");                                                                                                                                                            
  for (uint64_t i = 0; i < exe_time; ++i) {                  
    callSpFmaWithRegAD<GMMA_t, 1, RegCount>(gmma_instance, reg_a, std::make_index_sequence<1>{}, desc_b.desc_, reg_d, std::make_index_sequence<RegCount>{}, metaE);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
  startClk[blockIdx.x * blockDim.x + threadIdx.x] = start;
  stopClk[blockIdx.x * blockDim.x + threadIdx.x] = stop;
}

template<int N>
void test_m64nNk16_fmix_SS(std::string init_method) {

  int mat_a_size = M * K;
  int mat_b_size = N * K;
  int mat_c_size = M * N;

  __half *mat_a_host = new __half[mat_a_size];
  fill_mat(mat_a_host, mat_a_size, init_method);

  __half *mat_b_host = new __half[mat_b_size];
  fill_mat(mat_b_host, mat_b_size, init_method);

  float *mat_c_host = new float[mat_c_size];
  fill_mat(mat_c_host, mat_c_size, init_method);

  float *mat_d_host = new float[mat_c_size];
  fill_mat(mat_d_host, mat_c_size, init_method);

  __half2 *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__half)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__half),
                       cudaMemcpyHostToDevice));

  __half2 *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__half)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__half),
                       cudaMemcpyHostToDevice));

  float *mat_c_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(mat_c_dev, mat_c_host, mat_c_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  float *mat_d_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_c_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(mat_d_dev, mat_d_host, mat_c_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64nNk16_fmix_SS<N>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);
  int num_sm = 1;
  uint64_t *startClk = (uint64_t *)malloc(128 * num_sm * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(128 * num_sm * sizeof(uint64_t));
  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  gpuErrchk(cudaMalloc(&startClk_g, 128 * num_sm * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, 128 * num_sm * sizeof(uint64_t)));


  GpuTimer timer;
  timer.Start();
  wgmma_m64nNk16_fmix_SS<N><<<num_sm, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, 100000, startClk_g, stopClk_g);
  timer.Stop();
  float elapsed_time = timer.Elapsed();
  gpuErrchk(cudaMemcpy(startClk, startClk_g, 128 * num_sm * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, 128 * num_sm * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  uint64_t total_clk_num =
        *std::max_element(&stopClk[0], &stopClk[128 * num_sm]) -
        *std::min_element(&startClk[0], &startClk[128 * num_sm]);
  double FLOPS = (double)2 * M * K * N * 100000 * num_sm;
  double TFLOPS = FLOPS / elapsed_time / 1000 / 1000 / 1000;
  double latency = (double)total_clk_num / 100000;

  std::cout << "SM90_SP_64xNx32_F32F16F16_SS " << init_method
	      << " A in reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << " elapsed_time: " << elapsed_time << "ms " << TFLOPS << "TFLOPS" << ","
        << " latancy=" << latency
        << std::endl;
}

template <int N>
__global__ void
wgmma_m64nNk16_fmix_RS(float *gm_d, __half2 *gm_a, __half2 *gm_b,         
                                      float *gm_c, uint8_t sm_layout, int exe_time,
                                      uint64_t *startClk, uint64_t *stopClk) {                        
  extern __shared__ char shem[];                                                  
  constexpr int RegCount = 64 * N / 128;                                   
  using GMMA_t = typename GMMA_Selector_RS<N>::type;  
  GMMA_t gmma_instance;
  typename GMMA_t::CRegisters reg_d;                                                         
  typename GMMA_t::ARegisters reg_a;   

  __half2 *shem_a = (__half2 *)shem;                                               
  for (int i = threadIdx.x; i < M * K / 2 / 2; i += blockDim.x) {                        
    shem_a[i] = gm_a[i];
  }                                                                                  
                                                                                     
  __half2 *shem_b = (__half2 *)(shem + sizeof(__half2) * M * K / 2 / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K / 2; i += blockDim.x) {                        
    shem_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                                                                                                       
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(__half2); i += 1) {                        
    reinterpret_cast<__half2&>(reg_a[i]) = shem_a[i];
  }                                                                                  
                                                                                     
  uint32_t sm_a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shem_a));        
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shem_b));        
                                                                                     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_b.base_offset_ = 0;                                                                                                                 
                                                                                     
  desc_b.leading_byte_offset_ = (8 * 8 * sizeof(__half)) >> 4;                       
                                                                                     
  desc_b.stride_byte_offset_ = (4 * 8 * 8 * sizeof(__half)) >> 4;         
  uint32_t metaE = 0x44444444;   
  uint64_t start = 0;
  uint64_t stop = 0;
  asm volatile("bar.sync 0;");  
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");                                                                                                                    
  for (uint64_t i = 0; i < exe_time; ++i) {                  
    callSpFmaWithRegAD<GMMA_t, 4, RegCount>(gmma_instance, reg_a, std::make_index_sequence<4>{}, desc_b.desc_, reg_d, std::make_index_sequence<RegCount>{}, metaE);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
  startClk[blockIdx.x * blockDim.x + threadIdx.x] = start;
  stopClk[blockIdx.x * blockDim.x + threadIdx.x] = stop;
}

template <int N>
void test_m64nNk16_fmix_RS(std::string init_method) {
  int mat_a_size = M * K / 2;
  int mat_b_size = N * K;
  int mat_c_size = M * N;

  __half *mat_a_host = new __half[mat_a_size];
  fill_mat(mat_a_host, mat_a_size, init_method);

  __half *mat_b_host = new __half[mat_b_size];
  fill_mat(mat_b_host, mat_b_size, init_method);

  float *mat_c_host = new float[mat_c_size];
  fill_mat(mat_c_host, mat_c_size, init_method);

  float *mat_d_host = new float[mat_c_size];
  fill_mat(mat_d_host, mat_c_size, init_method);

  __half2 *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__half)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__half),
                       cudaMemcpyHostToDevice));

  __half2 *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__half)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__half),
                       cudaMemcpyHostToDevice));

  float *mat_c_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_c_dev, mat_c_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(mat_c_dev, mat_c_host, mat_c_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  float *mat_d_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_d_dev, mat_c_size * sizeof(float)));
  gpuErrchk(cudaMemcpy(mat_d_dev, mat_d_host, mat_c_size * sizeof(float),
                       cudaMemcpyHostToDevice));

  int dyn_shared_size =
      mat_a_size * sizeof(__half) + mat_b_size * sizeof(__half);

  cudaFuncSetAttribute(wgmma_m64nNk16_fmix_RS<N>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  int num_sm = 1;
  uint64_t *startClk = (uint64_t *)malloc(128 * num_sm * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(128 * num_sm * sizeof(uint64_t));
  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  gpuErrchk(cudaMalloc(&startClk_g, 128 * num_sm * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, 128 * num_sm * sizeof(uint64_t)));

  GpuTimer timer;
  timer.Start();
  wgmma_m64nNk16_fmix_RS<N><<<num_sm, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, 100000, startClk_g, stopClk_g);
  timer.Stop();
  float elapsed_time = timer.Elapsed();
  gpuErrchk(cudaMemcpy(startClk, startClk_g, 128 * num_sm * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, 128 * num_sm * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  uint64_t total_clk_num =
        *std::max_element(&stopClk[0], &stopClk[128 * num_sm]) -
        *std::min_element(&startClk[0], &startClk[128 * num_sm]);
  double FLOPS = (double)2 * M * K * N * 100000 * num_sm;
  double TFLOPS = FLOPS / elapsed_time / 1000 / 1000 / 1000;
  double latency = (double)total_clk_num / 100000;
  std::cout << "SM90_SP_64xNx32_F32F16F16_RS " << init_method
	      << " A in reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << " elapsed_time: " << elapsed_time << "ms " << TFLOPS << "TFLOPS" << ","
        << " latancy=" << latency
        << std::endl;
}


int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << " Usage ./fp16 <random/zero> <8 16 32 64 128 256> <ss/rs>" << std::endl;
    return -1;
  }
  std::string init_method = std::string(argv[1]);
  int N = std::stoi(argv[2]);
  std::string a_scope = std::string(argv[3]);
  if (a_scope == "ss") {
    if (N == 256) {
      test_m64nNk16_fmix_SS<256>(init_method);
    } else if (N == 128) {
      test_m64nNk16_fmix_SS<128>(init_method);
    } else if (N == 64) {
      test_m64nNk16_fmix_SS<64>(init_method);
    } else if (N == 32) {
      test_m64nNk16_fmix_SS<32>(init_method);
    } else if (N == 16) {
      test_m64nNk16_fmix_SS<16>(init_method);
    } else if (N == 8) {
      test_m64nNk16_fmix_SS<8>(init_method);
    } else {
      std::cout << "Unimplemented value of N: " << N << std::endl;
    }
  } else if (a_scope == "rs") {
    if (N == 256) {
      test_m64nNk16_fmix_RS<256>(init_method);
    } else if (N == 128) {
      test_m64nNk16_fmix_RS<128>(init_method);
    } else if (N == 64) {
      test_m64nNk16_fmix_RS<64>(init_method);
    } else if (N == 32) {
      test_m64nNk16_fmix_RS<32>(init_method);
    } else if (N == 16) {
      test_m64nNk16_fmix_RS<16>(init_method);
    } else if (N == 8) {
      test_m64nNk16_fmix_RS<8>(init_method);
    } else {
      std::cout << "Unimplemented value of N: " << N << std::endl;
    }
  }
}

