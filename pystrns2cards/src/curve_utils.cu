#include "cuda_common.cuh"

namespace strns2cards {
namespace cuda {

namespace {

__device__ float gaussian_weight(float dist_sq, float inv_beta2) {
    return expf(-dist_sq * inv_beta2);
}

__device__ void apply_rotation(
    const float* R, 
    const float* local, 
    float* rotated
) {
    for (int j = 0; j < 3; ++j) {
        rotated[j] = R[j * 3 + 0] * local[0] +
                     R[j * 3 + 1] * local[1] +
                     R[j * 3 + 2] * local[2];
    }
}

__global__ void smooth_strands_kernel(
    int num_strands,
    int num_points,
    const float* __restrict__ strands,        // (N, V, 3)
    const float* __restrict__ arclength,      // (N, V)
    float* __restrict__ smoothed_strands,     // (N, V, 3)
    float beta
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_strands) return;

    auto idx3 = [&](int p) { return s * num_points * 3 + p * 3; };
    auto idx1 = [&](int p) { return s * num_points + p; };

    float inv_beta2 = 1.0f / (beta * beta);
    float cutoff = 3.0f * beta;  // Three-beta cutoff distance

    for (int i = 0; i < num_points; ++i) {
        float si = arclength[idx1(i)];
        float acc[3] = {0.f, 0.f, 0.f};
        float sum_w = 0.f;

        for (int j = 0; j < num_points; ++j) {
            float sj = arclength[idx1(j)];
            float dist = fabsf(si - sj);

            if (dist > cutoff) continue;  // Skip distant points

            float dist_sq = dist * dist;
            float w = gaussian_weight(dist_sq, inv_beta2);

            acc[0] += w * strands[idx3(j) + 0];
            acc[1] += w * strands[idx3(j) + 1];
            acc[2] += w * strands[idx3(j) + 2];
            sum_w += w;
        }

        // Normalize the weighted sum
        if (sum_w > 1e-6f) {
            smoothed_strands[idx3(i) + 0] = acc[0] / sum_w;
            smoothed_strands[idx3(i) + 1] = acc[1] / sum_w;
            smoothed_strands[idx3(i) + 2] = acc[2] / sum_w;
        }
    }
}

__global__ void skinning_kernel(
    int num_strands,
    int num_points,
    const float* __restrict__ strands,            // (N, V, 3)
    const float* __restrict__ arclength,          // (N, V)
    const float* __restrict__ handle_strands,     // (N, V, 3)
    const float* __restrict__ canonical_handles,  // (N, V, 3)
    const float* __restrict__ R,                  // (N, V, 3, 3)
    float* __restrict__ skinned_strands,          // (N, V, 3)
    float beta
) {
    int strand_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (strand_idx >= num_strands) return;

    const float inv_beta2 = 1.0f / (beta * beta);
    const float cutoff = 3.0f * beta;  // Three-beta cutoff

    auto idx3 = [&](int i, int j) { return ((strand_idx * num_points + i) * 3 + j); };
    auto idx9 = [&](int i) { return ((strand_idx * num_points + i) * 9); };
    auto arc = [&](int i) { return arclength[strand_idx * num_points + i]; };

    for (int i = 0; i < num_points; ++i) {
        float si = arc(i);
        float acc[3] = {0.0f, 0.0f, 0.0f};
        float sum_w = 0.0f;

        for (int j = 0; j < num_points; ++j) {
            float sj = arc(j);
            float dist = fabsf(si - sj);

            if (dist > cutoff) continue;  // Apply three-beta cutoff

            float dist_sq = dist * dist;
            float w = gaussian_weight(dist_sq, inv_beta2);

            float local[3] = {
                strands[idx3(i, 0)] - handle_strands[idx3(j, 0)],
                strands[idx3(i, 1)] - handle_strands[idx3(j, 1)],
                strands[idx3(i, 2)] - handle_strands[idx3(j, 2)]
            };

            float rotated[3];
            apply_rotation(&R[idx9(j)], local, rotated);

            acc[0] += w * (rotated[0] + canonical_handles[idx3(j, 0)]);
            acc[1] += w * (rotated[1] + canonical_handles[idx3(j, 1)]);
            acc[2] += w * (rotated[2] + canonical_handles[idx3(j, 2)]);

            sum_w += w;
        }

        if (sum_w > 1e-8f) {
            skinned_strands[idx3(i, 0)] = acc[0] / sum_w;
            skinned_strands[idx3(i, 1)] = acc[1] / sum_w;
            skinned_strands[idx3(i, 2)] = acc[2] / sum_w;
        } else {
            skinned_strands[idx3(i, 0)] = strands[idx3(i, 0)];
            skinned_strands[idx3(i, 1)] = strands[idx3(i, 1)];
            skinned_strands[idx3(i, 2)] = strands[idx3(i, 2)];
        }
    }
}

} // unnamed namespace

void smooth_strands(
    int num_strands,
    int num_points,
    const float* strands,    // (N, V, 3)
    const float* arclength,  // (N, V, 3)
    float* smoothed_strands,
    float beta
) {
    smooth_strands_kernel<<<GET_BLOCKS(num_strands), THREADS_PER_BLOCK>>>(
        num_strands, num_points, strands, arclength, smoothed_strands, beta
    );
    CUDA_CALL(cudaGetLastError());
}

void skinning(
    int num_strands,
    int num_points,
    const float* strands,            // (N, V, 3)
    const float* arclength,          // (N, V)
    const float* handle_strands,     // (N, V, 3)
    const float* canonical_handles,  // (N, V, 3)
    const float* R,                  // (N, V, 3, 3)
    float* skinned_strands,          // (N, V, 3)
    float beta
) {
    skinning_kernel<<<GET_BLOCKS(num_strands), THREADS_PER_BLOCK>>>(
        num_strands, num_points,
        strands, arclength, handle_strands, canonical_handles, R,
        skinned_strands, beta
    );
    CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace strns2cards
