#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "mma_sm90_gmma.hpp"
using namespace cute;

constexpr int M = 64;
constexpr int K = 64;
template <int N>
struct GMMA_Selector_SS;

template <>
struct GMMA_Selector_SS<256> {
    using type = SM90_SP_64x256x64_F16E4M3E4M3_SS_TN<>;
};

template <int N>
struct GMMA_Selector_RS;

template <>
struct GMMA_Selector_RS<256> {
    using type = SM90_SP_64x256x64_F16E4M3E4M3_RS_TN<>;
};

template <int N>
__global__ void
wgmma_m64nNk32_e4m3_SS(float *gm_d, __nv_fp8_storage_t *gm_a, __nv_fp8_storage_t *gm_b,         
                                      float *gm_c, uint8_t sm_layout, int exe_time) {                        
  constexpr int RegCount = 64 * N / 128 / 2;
  extern __shared__ char shem[];                                                  
  using GMMA_t = typename GMMA_Selector_SS<N>::type;  
  GMMA_t gmma_instance;
  typename GMMA_t::CRegisters reg_d;                                                         
  typename GMMA_t::ARegisters reg_a;                                                             
                                                                                                                                                            
  __nv_fp8_storage_t *shem_a = (__nv_fp8_storage_t *)shem;                                               
  for (int i = threadIdx.x; i < M * K; i += blockDim.x) {                        
    shem_a[i] = gm_a[i];                                                               
  }                                                                                                                                                                     
  __nv_fp8_storage_t *shem_b = (__nv_fp8_storage_t *)(shem + sizeof(__nv_fp8_storage_t) * M * K);                
                                          
  for (int i = threadIdx.x; i < N * K; i += blockDim.x) {                        
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
                                                                                     
  desc_a.leading_byte_offset_ = (8 * 32 * sizeof(__nv_fp8_storage_t)) >> 4;                       
  desc_b.leading_byte_offset_ = (8 * 16 * sizeof(__nv_fp8_storage_t)) >> 4;                       
                                                                                     
  desc_a.stride_byte_offset_ = (2 * 8 * 32 * sizeof(__nv_fp8_storage_t)) >> 4;                    
  desc_b.stride_byte_offset_ = (4 * 8 * 16 * sizeof(__nv_fp8_storage_t)) >> 4;                    
                                                                                     
  reg_a[0] = desc_a.desc_;   
  uint32_t metaE = 0x44444444;                                                                                      
  for (uint64_t i = 0; i < exe_time; ++i) {                  
    callSpFmaWithRegAD<GMMA_t, 1, RegCount>(gmma_instance, reg_a, std::make_index_sequence<1>{}, desc_b.desc_, reg_d, std::make_index_sequence<RegCount>{}, metaE);
  }

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}
template <int N>
void test_m64nNk32_e4m3_SS(std::string init_method) {
  int mat_a_size = M * K;
  int mat_b_size = N * K;
  int mat_c_size = M * N / 2;

  __nv_fp8_storage_t *mat_a_host = new __nv_fp8_storage_t[mat_a_size];
  fill_mat(mat_a_host, mat_a_size, init_method);

  __nv_fp8_storage_t *mat_b_host = new __nv_fp8_storage_t[mat_b_size];
  fill_mat(mat_b_host, mat_b_size, init_method);

  float *mat_c_host = new float[mat_c_size];
  fill_mat(mat_c_host, mat_c_size, init_method);

  float *mat_d_host = new float[mat_c_size];
  fill_mat(mat_d_host, mat_c_size, init_method);

  __nv_fp8_storage_t *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__nv_fp8_storage_t)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__nv_fp8_storage_t),
                       cudaMemcpyHostToDevice));

  __nv_fp8_storage_t *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__nv_fp8_storage_t)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__nv_fp8_storage_t),
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
      mat_a_size * sizeof(__nv_fp8_storage_t) + mat_b_size * sizeof(__nv_fp8_storage_t);

  cudaFuncSetAttribute(wgmma_m64nNk32_e4m3_SS<N>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sm = prop.multiProcessorCount;

  GpuTimer timer;
  timer.Start();
  wgmma_m64nNk32_e4m3_SS<N><<<num_sm, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, 100000);
  timer.Stop();
  float elapsed_time = timer.Elapsed();

  double FLOPS = (double)2 * M * K * N * 100000 * num_sm;
  double TFLOPS = FLOPS / elapsed_time / 1000 / 1000 / 1000;

  std::cout << "SM90_SP_64xNx64_F16E4M3E4M3_SS_TN " << init_method
	      << " A in shem "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  elapsed_time: " << elapsed_time << "ms " << TFLOPS << "TFLOPS"
        << std::endl;
}

template <int N>
__global__ void
wgmma_m64nNk32_e4m3_RS(float *gm_d, __nv_fp8_storage_t *gm_a, __nv_fp8_storage_t *gm_b,         
                                      float *gm_c, uint8_t sm_layout, int exe_time) {                        

  extern __shared__ char shem[];                                                  
  constexpr int RegCount = 64 * N / 128 / 2;                                   
  using GMMA_t = typename GMMA_Selector_RS<N>::type;  
  GMMA_t gmma_instance;
  typename GMMA_t::CRegisters reg_d;                                                         
  typename GMMA_t::ARegisters reg_a;                                                             
                                                                                    
  __nv_fp8_storage_t *shem_a = (__nv_fp8_storage_t *)shem;                                               
  for (int i = threadIdx.x; i < M * K / 2; i += blockDim.x) {                        
    shem_a[i] = gm_a[i];                                                               
  }                                                                                  
                                                                                     
  __nv_fp8_storage_t *shem_b = (__nv_fp8_storage_t *)(shem + sizeof(__nv_fp8_storage_t) * M * K / 2);                
                                                                                     
  for (int i = threadIdx.x; i < N * K; i += blockDim.x) {                        
    shem_b[i] = gm_b[i];                                                               
  }                                                                                  
                                                                                     
  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {                            
    reg_d[i] = gm_c[threadIdx.x + blockDim.x * i];                                   
  }                                                                                  
                                                                                     
  __syncthreads();                                                                   

  for (int i = 0; i < sizeof(reg_a)/sizeof(uint32_t); i += 1) {
    reg_a[i] = ((uint32_t*)shem_a)[i];
  }
                                                                                     
  uint32_t sm_b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shem_b));        
                                                                                     
  GmmaDescriptor desc_b;                                                     
  desc_b.layout_type_ = sm_layout;                                                   
                                                                                     
  desc_b.start_address_ = sm_b_addr >> 4;                                            
                                                                                     
  desc_b.base_offset_ = 0;                                                           
                                                                                     
  desc_b.leading_byte_offset_ = (8 * 16 * sizeof(__nv_fp8_storage_t)) >> 4;                       
                                                                                     
  desc_b.stride_byte_offset_ = (4 * 8 * 16 * sizeof(__nv_fp8_storage_t)) >> 4;                    
  uint32_t metaE = 0x44444444;                                                                                      
                                                                                     
  for (uint64_t i = 0; i < exe_time; ++i) {                  
    callSpFmaWithRegAD<GMMA_t, 4, RegCount>(gmma_instance, reg_a, std::make_index_sequence<4>{}, desc_b.desc_, reg_d, std::make_index_sequence<RegCount>{},metaE);
  }
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(0) : "memory");

  for (int i = 0; i < sizeof(reg_d)/sizeof(float); ++i) {
    gm_d[threadIdx.x + blockDim.x * i] = reg_d[i];
  }
}

template <int N>
void test_m64nNk32_e4m3_RS(std::string init_method) {
  int mat_a_size = M * K / 2;
  int mat_b_size = N * K;
  int mat_c_size = M * N / 2;

  __nv_fp8_storage_t *mat_a_host = new __nv_fp8_storage_t[mat_a_size];
  fill_mat(mat_a_host, mat_a_size, init_method);

  __nv_fp8_storage_t *mat_b_host = new __nv_fp8_storage_t[mat_b_size];
  fill_mat(mat_b_host, mat_b_size, init_method);

  float *mat_c_host = new float[mat_c_size];
  fill_mat(mat_c_host, mat_c_size, init_method);

  float *mat_d_host = new float[mat_c_size];
  fill_mat(mat_d_host, mat_c_size, init_method);

  __nv_fp8_storage_t *mat_a_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_a_dev, mat_a_size * sizeof(__nv_fp8_storage_t)));
  gpuErrchk(cudaMemcpy(mat_a_dev, mat_a_host, mat_a_size * sizeof(__nv_fp8_storage_t),
                       cudaMemcpyHostToDevice));

  __nv_fp8_storage_t *mat_b_dev = nullptr;
  gpuErrchk(cudaMalloc(&mat_b_dev, mat_b_size * sizeof(__nv_fp8_storage_t)));
  gpuErrchk(cudaMemcpy(mat_b_dev, mat_b_host, mat_b_size * sizeof(__nv_fp8_storage_t),
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
      mat_a_size * sizeof(__nv_fp8_storage_t) + mat_b_size * sizeof(__nv_fp8_storage_t);

  cudaFuncSetAttribute(wgmma_m64nNk32_e4m3_RS<N>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       dyn_shared_size);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sm = prop.multiProcessorCount;
  
  GpuTimer timer;
  timer.Start();
  wgmma_m64nNk32_e4m3_RS<N><<<num_sm, 128, dyn_shared_size>>>(
        mat_d_dev, mat_a_dev, mat_b_dev, mat_c_dev, 0, 100000);
  timer.Stop();
  float elapsed_time = timer.Elapsed();

  double FLOPS = (double)2 * M * K * N * 100000 * num_sm;
  double TFLOPS = FLOPS / elapsed_time / 1000 / 1000 / 1000;

  std::cout << "SM90_64xNx32_F16E4M3E4M3_RS_TN " << init_method
	      << " A in reg "
	      << "M=" << M << ","
	      << "N=" << N << ","
	      << "K=" << K << "  elapsed_time: " << elapsed_time << "ms " << TFLOPS << "TFLOPS"
        << std::endl;
 
}


int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << " Usage ./e4m3fp16 <random/zero> <256> <ss/rs>" << std::endl;
    return -1;
  }
  std::string init_method = std::string(argv[1]);
  int N = std::stoi(argv[2]);
  std::string a_scope = std::string(argv[3]);
  if (a_scope == "ss") {
    if (N == 256) {
      test_m64nNk32_e4m3_SS<256>(init_method);
    } else {
      std::cout << "Unimplemented value of N: " << N << std::endl;
    }
  } else if (a_scope == "rs") {
    if (N == 256) {
      test_m64nNk32_e4m3_RS<256>(init_method);
    } else {
      std::cout << "Unimplemented value of N: " << N << std::endl;
    }
  }
}

