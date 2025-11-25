#include "cuda_common.cuh"

namespace strns2cards {
namespace cuda {

__global__ void accumulate_bitset_rgb_kernel(
    int B, int H, int W, int num_slots,
    const int* bitset,                  // [B * H * W * num_slots]
    const float* cluster_colors,        // [num_clusters, 3]
    float* color_out                    // [B * H * W * 3]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = B * H * W;
    if (idx >= total_pixels) return;

    float r = 0.f, g = 0.f, b = 0.f;

    for (int s = 0; s < num_slots; ++s) {
        const int packed = bitset[idx * num_slots + s];
        if (packed == 0) continue;

        for (int j = 0; j < 32; ++j) {
            if (packed & (1 << j)) {
                const int cluster_id = s * 32 + j;
                r += cluster_colors[cluster_id * 3 + 0];
                g += cluster_colors[cluster_id * 3 + 1];
                b += cluster_colors[cluster_id * 3 + 2];
            }
        }
    }

    color_out[idx * 3 + 0] = r;
    color_out[idx * 3 + 1] = g;
    color_out[idx * 3 + 2] = b;
}

__global__ void popcount_bitset_kernel(
    int B, int H, int W, int num_slots,
    const int* bitset,              // [B * H * W * num_slots]
    float* weight_out               // [B * H * W]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = B * H * W;
    if (idx >= total_pixels) return;

    int count = 0;
    for (int s = 0; s < num_slots; ++s) {
        const int packed = bitset[idx * num_slots + s];
        count += __popc(packed);  // fast hardware popcount
    }
    weight_out[idx] = static_cast<float>(count);
}

// --- Launcher Functions ---

void accumulate_bitset_rgb(
    int B, int H, int W, int num_slots,
    const int* bitset,              // [B * H * W * num_slots]
    const float* cluster_rgb,       // [num_clusters, 3]
    float* accum_color              // [B * H * W * 3]
) {
    const int total_pixels = B * H * W;
    accumulate_bitset_rgb_kernel<<<GET_BLOCKS(total_pixels), THREADS_PER_BLOCK>>>(
        B, H, W, num_slots,
        bitset,
        cluster_rgb,
        accum_color
    );
    CUDA_CALL(cudaGetLastError());
}

void popcount_bitset(
    int B, int H, int W, int num_slots,
    const int* bitset,     // [B * H * W * num_slots]
    float* count_image     // [B * H * W]
) {
    const int total_pixels = B * H * W;
    popcount_bitset_kernel<<<GET_BLOCKS(total_pixels), THREADS_PER_BLOCK>>>(
        B, H, W, num_slots,
        bitset,
        count_image
    );
    CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace strns2cards
