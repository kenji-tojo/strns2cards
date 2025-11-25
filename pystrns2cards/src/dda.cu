#include "cuda_common.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace strns2cards {
namespace cuda {

__global__ void dda_compute_span_kernel(
    const int num_prims, const int *edge_ids,
    const int H, const int W, const int V,
    const float *pos, const int E, const int *edges,
    int *frag_counts, float *frag_slopes, float *frag_spans
) {
    const unsigned int prim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (prim_idx >= num_prims) return;

    const int batch_idx = edge_ids[prim_idx] / E;
    const int edge_idx  = edge_ids[prim_idx] % E;

    const int v0_idx = edges[edge_idx * 2 + 0];
    const int v1_idx = edges[edge_idx * 2 + 1];
    const float *const pos_ptr = pos + batch_idx * V * 4;

    const float x0_ndc = pos_ptr[v0_idx * 4 + 0] / pos_ptr[v0_idx * 4 + 3];
    const float y0_ndc = pos_ptr[v0_idx * 4 + 1] / pos_ptr[v0_idx * 4 + 3];
    const float x1_ndc = pos_ptr[v1_idx * 4 + 0] / pos_ptr[v1_idx * 4 + 3];
    const float y1_ndc = pos_ptr[v1_idx * 4 + 1] / pos_ptr[v1_idx * 4 + 3];

    const float fH = static_cast<float>(H);
    const float fW = static_cast<float>(W);

    float x0 = 0.5f * (x0_ndc + 1.f) * fW;
    float y0 = 0.5f * (1.f - y0_ndc) * fH;
    float x1 = 0.5f * (x1_ndc + 1.f) * fW;
    float y1 = 0.5f * (1.f - y1_ndc) * fH;

    const float delta_x = x1 - x0;
    const float delta_y = y1 - y0;

    if (delta_x == 0.f && delta_y == 0.f) {
        frag_counts[prim_idx] = 0;
        return;
    }

    if (fabsf(delta_x) >= fabsf(delta_y)) {
        const float slope_val = delta_y / delta_x;
        frag_slopes[prim_idx * 2 + 0] = 0.f; // horizontal sweep
        frag_slopes[prim_idx * 2 + 1] = slope_val;

        if (x0 > x1) {
            device_swap(x0, x1);
            device_swap(y0, y1);
        }

        const float frag_start_x = MAX(floorf(x0), 0.f);
        const float frag_start_y = y0 + slope_val * (frag_start_x - x0);
        const int x_start_idx = static_cast<int>(frag_start_x);
        const int x_end_idx   = MIN(static_cast<int>(floorf(x1)), W - 1) + 1;
        frag_counts[prim_idx] = MAX(x_end_idx - x_start_idx, 0);

        frag_spans[prim_idx * 4 + 0] = frag_start_x;
        frag_spans[prim_idx * 4 + 1] = frag_start_y;
        frag_spans[prim_idx * 4 + 2] = x0 - frag_start_x;
        frag_spans[prim_idx * 4 + 3] = x1 - frag_start_x;
    } else {
        const float slope_val = delta_x / delta_y;
        frag_slopes[prim_idx * 2 + 0] = 1.f; // vertical sweep
        frag_slopes[prim_idx * 2 + 1] = slope_val;

        if (y0 > y1) {
            device_swap(x0, x1);
            device_swap(y0, y1);
        }

        const float frag_start_y = MAX(floorf(y0), 0.f);
        const float frag_start_x = x0 + slope_val * (frag_start_y - y0);
        const int y_start_idx = static_cast<int>(frag_start_y);
        const int y_end_idx   = MIN(static_cast<int>(floorf(y1)), H - 1) + 1;
        frag_counts[prim_idx] = MAX(y_end_idx - y_start_idx, 0);

        frag_spans[prim_idx * 4 + 0] = frag_start_x;
        frag_spans[prim_idx * 4 + 1] = frag_start_y;
        frag_spans[prim_idx * 4 + 2] = y0 - frag_start_y;
        frag_spans[prim_idx * 4 + 3] = y1 - frag_start_y;
    }
}

int dda_compute_span(
    int num_prims,
    const int* edge_ids,         // [num_prims]
    int H, int W, int V,
    const float* pos,            // [B * V * 4]
    int E,
    const int* edges,            // [E * 2]
    int* frag_prefix_sum,        // [num_prims]
    float* frag_slopes,          // [num_prims, 2]
    float* frag_spans            // [num_prims, 4]
) {
    int* frag_counts = nullptr;
    CUDA_CALL(cudaMalloc(&frag_counts, sizeof(int) * num_prims));

    // Launch kernel to compute span and frag count
    dda_compute_span_kernel<<<GET_BLOCKS(num_prims), THREADS_PER_BLOCK>>>(
        num_prims, edge_ids, H, W, V, pos, E, edges,
        frag_counts, frag_slopes, frag_spans
    );
    CUDA_CALL(cudaGetLastError());

    // Inclusive scan to get prefix sum
    thrust::inclusive_scan(
        thrust::device_pointer_cast(frag_counts),
        thrust::device_pointer_cast(frag_counts + num_prims),
        thrust::device_pointer_cast(frag_prefix_sum)
    );

    // Copy total fragment count to host
    int num_frags = 0;
    CUDA_CALL(cudaMemcpy(&num_frags, frag_prefix_sum + (num_prims - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(frag_counts));
    return num_frags;
}

__device__ __forceinline__ float perspective_correct_interpolate(
    int H, int W, const float p0[4], const float p1[4], float x, float y, float axis
) {
    float t0, t1;
    if (axis == 0.f) {
        float qx = -1.f + 2.f * (x / static_cast<float>(W));
        float p0x = p0[0] - qx * p0[3];
        float p1x = p1[0] - qx * p1[3];
        t0 = fabsf(p1x);
        t1 = fabsf(p0x);
    } else {
        float qy = -1.f + 2.f * (y / static_cast<float>(H));
        float p0y = -p0[1] - qy * p0[3];
        float p1y = -p1[1] - qy * p1[3];
        t0 = fabsf(p1y);
        t1 = fabsf(p0y);
    }
    float denom = t0 + t1;
    if (denom < 1e-6f) return 0.5f;
    return t1 / denom;
}

__global__ void dda_compute_fragments_kernel(
    int num_prims,
    int num_frags,
    const int *frag_prefix_sum,       // [num_prims]
    const int *edge_ids,              // [num_prims]
    const float *frag_slopes,         // [num_prims, 2]
    const float *frag_spans,          // [num_prims, 4]
    int H, int W, int V,
    const float *pos,                 // [B * V * 4]
    int E,
    const int *edges,                 // [E * 2]
    int *frag_pix,                    // [num_frags, 3]
    float *frag_attrs                 // [num_frags, 4]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    // Binary search to find primitive index
    int lo = 0, hi = num_prims;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (frag_prefix_sum[mid] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    const int prim_idx = lo;
    const int batch_idx = edge_ids[prim_idx] / E;
    const int edge_idx  = edge_ids[prim_idx] % E;

    const float start_x = frag_spans[prim_idx * 4 + 0];
    const float start_y = frag_spans[prim_idx * 4 + 1];
    const float offset0 = frag_spans[prim_idx * 4 + 2];
    const float offset1 = frag_spans[prim_idx * 4 + 3];

    const int frag_base = (prim_idx == 0) ? 0 : frag_prefix_sum[prim_idx - 1];
    const int frag_idx_in_prim = static_cast<int>(idx) - frag_base;
    const float d = fminf(fmaxf(static_cast<float>(frag_idx_in_prim) + 0.5f, offset0), offset1);

    const float axis = frag_slopes[prim_idx * 2 + 0];
    const float slope_val = frag_slopes[prim_idx * 2 + 1];

    const float delta_x = (axis == 0.f) ? d : d * slope_val;
    const float delta_y = (axis == 0.f) ? d * slope_val : d;
    const float x = start_x + delta_x;
    const float y = start_y + delta_y;

    const int h = static_cast<int>(floorf(y));
    const int w = static_cast<int>(floorf(x));
    if (h < 0 || h >= H || w < 0 || w >= W) return;

    const int v0_idx = edges[edge_idx * 2 + 0];
    const int v1_idx = edges[edge_idx * 2 + 1];
    const float *pos_ptr = &pos[batch_idx * V * 4];

    float p0[4], p1[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        p0[i] = pos_ptr[v0_idx * 4 + i];
        p1[i] = pos_ptr[v1_idx * 4 + i];
    }

    const float t1 = perspective_correct_interpolate(H, W, p0, p1, x, y, axis);
    const float t0 = 1.f - t1;
    const float pz = t0 * p0[2] + t1 * p1[2];
    const float pw = t0 * p0[3] + t1 * p1[3];
    const float zw = pz / pw;

    if (!isfinite(zw) || zw < -1.f || zw > 1.f) return;

    frag_pix[idx * 3 + 0] = batch_idx;
    frag_pix[idx * 3 + 1] = h;
    frag_pix[idx * 3 + 2] = w;
    frag_attrs[idx * 4 + 0] = axis;
    frag_attrs[idx * 4 + 1] = t1;
    frag_attrs[idx * 4 + 2] = zw;
    frag_attrs[idx * 4 + 3] = static_cast<float>(edge_idx + 1);
}

void dda_compute_fragments(
    int num_prims,
    int num_frags,
    const int *frag_prefix_sum,
    const int *edge_ids,
    const float *frag_slopes,
    const float *frag_spans,
    int H, int W, int V,
    const float *pos,
    int E,
    const int *edges,
    int *frag_pix,
    float *frag_attrs
) {
    dda_compute_fragments_kernel<<<GET_BLOCKS(num_frags), THREADS_PER_BLOCK>>>(
        num_prims,
        num_frags,
        frag_prefix_sum,
        edge_ids,
        frag_slopes,
        frag_spans,
        H, W, V,
        pos,
        E,
        edges,
        frag_pix,
        frag_attrs
    );
    CUDA_CALL(cudaGetLastError());
}

__global__ void dda_interpolate_attributes_kernel(
    int B, int H, int W,
    const float* __restrict__ rast_dda,   // [B * H * W, 4]
    int C,
    const float* __restrict__ attr,       // [V, C]
    const int* __restrict__ edges,        // [E, 2]
    float* __restrict__ image             // [B * H * W, C]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const float t1 = rast_dda[idx * 4 + 1];
    const float t0 = 1.f - t1;
    const int edge_idx = static_cast<int>(rast_dda[idx * 4 + 3]) - 1;

    if (edge_idx < 0) return;

    const int v0_idx = edges[edge_idx * 2 + 0];
    const int v1_idx = edges[edge_idx * 2 + 1];

    const float* a0 = attr + v0_idx * C;
    const float* a1 = attr + v1_idx * C;

    for (int i = 0; i < C; ++i) {
        image[idx * C + i] = t0 * a0[i] + t1 * a1[i];
    }
}

void dda_interpolate_attributes(
    int B, int H, int W, int C,
    const float* rast_dda,    // [B * H * W, 4]
    const float* attr,        // [V, C]
    const int* edges,         // [E, 2]
    float* image              // [B * H * W, C]
) {
    const int num_pixels = B * H * W;
    dda_interpolate_attributes_kernel<<<GET_BLOCKS(num_pixels), THREADS_PER_BLOCK>>>(
        B, H, W, rast_dda, C, attr, edges, image
    );
    CUDA_CALL(cudaGetLastError());
}

__global__ void backward_dda_interpolate_attributes_kernel(
    int B, int H, int W,
    const float* rast_dda,           // [B, H, W, 4]
    int C,
    const float* d_image,            // [B, H, W, C]
    const int* edges,                // [E, 2]
    float* d_attr                    // [V, C]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    const float* grad_out = d_image + C * idx;

    const float t1 = rast_dda[idx * 4 + 1];
    const float t0 = 1.f - t1;
    const int edge_idx = static_cast<int>(rast_dda[idx * 4 + 3]) - 1;
    if (edge_idx < 0) return;

    const int v0 = edges[edge_idx * 2 + 0];
    const int v1 = edges[edge_idx * 2 + 1];
    float* d0 = d_attr + C * v0;
    float* d1 = d_attr + C * v1;

    for (int i = 0; i < C; ++i) {
        atomicAdd(&d0[i], grad_out[i] * t0);
        atomicAdd(&d1[i], grad_out[i] * t1);
    }
}

void backward_dda_interpolate_attributes(
    int B, int H, int W,
    const float* rast_dda,
    int C,
    const float* d_image,
    const int* edges,
    float* d_attr
) {
    const int total_pixels = B * H * W;
    backward_dda_interpolate_attributes_kernel<<<GET_BLOCKS(total_pixels), THREADS_PER_BLOCK>>>(
        B, H, W, rast_dda, C, d_image, edges, d_attr
    );
    CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace strns2cards
