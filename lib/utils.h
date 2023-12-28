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

#define sync_and_check_cuda_error() \
    _sync_and_check_cuda_error(__FILE__, __LINE__)
