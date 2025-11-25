#pragma once

// --- Includes ---
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <cstdint>

// --- Useful macros ---
#define CUDA_CALL(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); }} while (0)

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

// --- Launch bounds helper ---
#define THREADS_PER_BLOCK 256
#define GET_BLOCKS(N) ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

template <typename T>
__device__ __forceinline__ void device_swap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}
