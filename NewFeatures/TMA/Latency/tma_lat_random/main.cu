#include "MeasurementSeries.hpp"
#include "dtime.hpp"
//#include "gpu-clock.cuh"
#include "gpu-error.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>
#include <sys/time.h>
#include <cuda/barrier>
#include <nvml.h>
__global__ void powerKernel(double *A, int iters)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    double start = A[0];
#pragma unroll 1
    for (int i = 0; i < iters; i++)
    {
        start -= (tidx * 0.1) * start;
    }
    A[0] = start;
}

unsigned int getGPUClock(int deviceId)
{

    double *dA = NULL;
    GPU_ERROR(cudaMalloc(&dA, sizeof(double)));


    unsigned int gpu_clock;

    int iters = 10;

    powerKernel<<<1000, 1024>>>(dA, iters);

    double dt = 0;
    std::cout << "clock: ";
    while (dt < 0.4)
    {
        GPU_ERROR(cudaDeviceSynchronize());
        double t1 = dtime();

        powerKernel<<<1000, 1024>>>(dA, iters);
        usleep(10000);

        nvmlInit();
        nvmlDevice_t device;
        GPU_ERROR(cudaGetDevice(&deviceId));
        nvmlDeviceGetHandleByIndex(deviceId, &device);
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &gpu_clock);
        GPU_ERROR(cudaDeviceSynchronize());

        double t2 = dtime();
        std::cout << gpu_clock << " ";
        std::cout.flush();
        dt = t2 - t1;
        iters *= 2;
    }
    std::cout << "\n";
    GPU_ERROR(cudaFree(dA));
    return gpu_clock;
}


#define LOAD_SIZE 16 // bytes
typedef int64_t dtype;
const int unroll_factor = 32;


using namespace std;
using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ unsigned int smid()
{
    unsigned int r;

    asm("mov.u32 %0, %%smid;" : "=r"(r));

    return r;
}

template <typename T>
__global__ void pchase(T *buf, T *__restrict__ dummy_buf, int64_t N)
{

    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t uid = blockIdx.x * blockDim.x + tid;

    
    __shared__ alignas(16) uint64_t ptr[LOAD_SIZE / sizeof(dtype)];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (uid == 0) {

        init(&bar, blockDim.x);                    // a)
        asm volatile("fence.proxy.async.shared::cta;");     // b)
        ptr[0] = reinterpret_cast<uint64_t>(buf);

        #pragma unroll 1
        for (int64_t n = 0; n < N; n += unroll_factor)
        {
        #pragma unroll
            for (int u = 0; u < unroll_factor; u++)
            {
                asm volatile(
                    "{\t\n"
                    //"discard.L2 [%1], 128;\n\t"
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes[%0], [%1], %2, [%3]; // 1a. unicast\n\t"
                    "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%3], %2;\n\t"
                    "}"
                    :
                    : "r"(static_cast<unsigned>(__cvta_generic_to_shared(ptr))), "l"(ptr[0]), "n"(LOAD_SIZE), "r"(static_cast<unsigned>(__cvta_generic_to_shared(&bar)))
                    : "memory"); 


                // 3b. All threads arrive on the barrier
                barrier::arrival_token token = bar.arrive();

                // 3c. Wait for the data to have arrived.
                bar.wait(std::move(token));
                //asm volatile("fence.proxy.async.shared::cta;");
            }
        }
    }


    if (tid > 12313)
    {
        dummy_buf[0] = ptr[0];
    }
}

int main(int argc, char **argv)
{
    int device = 0;
    if (argc > 1)
    {
        device = atoi(argv[1]);
        cudaSetDevice(device);
    }
    unsigned int clock = getGPUClock(device);

    const int cl_size = 1;
    const int skip_factor = 16;

    for (int64_t LEN = 16; LEN < (1 << 24); LEN = LEN * 1.04 + 32)
    {
        if (LEN * skip_factor * cl_size * sizeof(dtype) > 120 * 1024 * 1024)
            LEN *= 1.5;

        const int64_t iters = max(LEN, (int64_t)1000000);
        // const int64_t iters =
        //     max((int64_t)2, ((int64_t)1 << 19) / LEN) * LEN * cl_size;

        vector<int64_t> order(LEN);
        int64_t *buf = NULL;
        int64_t *dbuf = NULL;
        dtype *dummy_buf = NULL;

        GPU_ERROR(cudaMallocManaged(&buf, skip_factor * cl_size * LEN * sizeof(dtype)));
        GPU_ERROR(cudaMalloc(&dbuf, skip_factor * cl_size * LEN * sizeof(dtype)));
        GPU_ERROR(cudaMallocManaged(&dummy_buf, sizeof(dtype)));
        for (int64_t i = 0; i < LEN; i++)
        {
            order[i] = i + 1;
        }
        order[LEN - 1] = 0;

        std::random_device rd;
        std::mt19937 g(rd());
        shuffle(begin(order), end(order) - 1, g);

        for (int cl_lane = 0; cl_lane < cl_size; cl_lane++)
        {
            dtype idx = 0;
            for (int64_t i = 0; i < LEN; i++)
            {

                buf[(idx * cl_size + cl_lane) * skip_factor] =
                    skip_factor *
                    (order[i] * cl_size + cl_lane + (order[i] == 0 ? 1 : 0));
                idx = order[i];
            }
        }
        buf[skip_factor * (order[LEN - 2] * cl_size + cl_size - 1)] = 0;

        for (int64_t n = 0; n < LEN * cl_size * skip_factor; n++)
        {
            buf[n] = (int64_t)dbuf + buf[n] * sizeof(int64_t *);
        }

        cudaMemcpy(dbuf, buf, skip_factor * cl_size * LEN * sizeof(dtype),
                   cudaMemcpyHostToDevice);

        pchase<dtype><<<1, 1>>>(dbuf, dummy_buf, iters);
        GPU_ERROR(cudaDeviceSynchronize());
        MeasurementSeries times;
        for (int i = 0; i < 7; i++)
        {
            GPU_ERROR(cudaDeviceSynchronize());
            double start = dtime();
            pchase<dtype><<<1, 1>>>(dbuf, dummy_buf, iters);
            GPU_ERROR(cudaDeviceSynchronize());
            double end = dtime();
            times.add(end - start);
        }

        GPU_ERROR(cudaGetLastError());

        double dt = times.minValue();
        cout << setw(9) << iters << " " << setw(5) << clock << " " //
             << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024
             << " "                                            //
             << fixed                                          //
             << setprecision(1) << setw(8) << dt * 1000 << " " //
             << setw(7) << setprecision(1)
             << (double)dt / iters * clock * 1000 * 1000 << "\n"
             << flush;

        GPU_ERROR(cudaFree(buf));
        GPU_ERROR(cudaFree(dbuf));
        GPU_ERROR(cudaFree(dummy_buf));
    }
    cout << "\n";
}
