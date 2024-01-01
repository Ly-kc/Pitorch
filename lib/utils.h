#pragma once

#include <cuda_runtime_api.h>

inline void sync_and_check_cuda_error_force() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

inline void _sync_and_check_cuda_error(const char* file, int line) {
#ifdef CHECK_CUDA_ERROR
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s (%d) %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

#define sync_and_check_cuda_error() \
    _sync_and_check_cuda_error(__FILE__, __LINE__)
