#include "cuda_common.cuh"

namespace strns2cards {
namespace cuda {

// --- Filter and compact valid entries in a 1D int array
__global__ void compact_valid_ints_kernel(
    int N,
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ global_counter
) {
    extern __shared__ int shared_scan[]; // shared memory
    int* valid_flags = shared_scan;
    int* block_base  = shared_scan + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Step 1: Predicate check
    int is_valid = 0;
    if (idx < N) {
        is_valid = (input[idx] >= 0);
    }
    valid_flags[tid] = is_valid;
    __syncthreads();

    // Step 2: Inclusive scan (Hillis–Steele)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset)
            temp = valid_flags[tid - offset];
        __syncthreads();
        valid_flags[tid] += temp;
        __syncthreads();
    }

    // Step 3: Block-level total
    if (tid == blockDim.x - 1) {
        int total = valid_flags[tid];
        block_base[0] = atomicAdd(global_counter, total);
    }
    __syncthreads();

    // Step 4: Write outputs
    if (idx < N && is_valid) {
        int local_idx = (tid > 0) ? valid_flags[tid - 1] : 0;
        int output_idx = block_base[0] + local_idx;
        output[output_idx] = input[idx];
    }
}

// --- Host launcher
int compact_valid_ints(
    int N,
    const int* input,     // [N]
    int* output           // [N] (preallocated)
) {
    if (N == 0) return 0;

    int* global_counter = nullptr;
    CUDA_CALL(cudaMalloc(&global_counter, sizeof(int)));
    CUDA_CALL(cudaMemset(global_counter, 0, sizeof(int)));

    const size_t shared_mem = sizeof(int) * (THREADS_PER_BLOCK + 1);

    compact_valid_ints_kernel<<<GET_BLOCKS(N), THREADS_PER_BLOCK, shared_mem>>>(
        N, input, output, global_counter
    );

    int valid_count = 0;
    CUDA_CALL(cudaMemcpy(&valid_count, global_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(global_counter));
    CUDA_CALL(cudaGetLastError()); // catch kernel launch errors

    return valid_count;
}

} // namespace cuda
} // namespace strns2cards
