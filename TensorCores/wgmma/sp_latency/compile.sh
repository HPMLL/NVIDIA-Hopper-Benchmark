 nvcc test_wgmma_fp16.cu -gencode arch=compute_90a,code=sm_90a  -I ../ -o fp16_sp_lat
 nvcc test_wgmma_fp16fp16.cu -gencode arch=compute_90a,code=sm_90a  -I ../ -o fp16fp16_sp_lat
 nvcc test_wgmma_tf32.cu -gencode arch=compute_90a,code=sm_90a  -I ../ -o tf32_sp_lat
 nvcc test_wgmma_e4m3.cu -gencode arch=compute_90a,code=sm_90a  -I ../ -o e4m3_sp_lat
 nvcc test_wgmma_fp16e4m3.cu -gencode arch=compute_90a,code=sm_90a  -I ../ -o fp16e4m3_sp_lat
 nvcc test_wgmma_s8.cu -gencode arch=compute_90a,code=sm_90a  -I ../ -o s8_sp_lat

