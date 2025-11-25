#include "cuda_common.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace strns2cards {
namespace cuda {

__global__ void compute_triangle_rects_kernel(
    int H, int W, int B,
    int V, const float *pos,      // [B * V][4]
    int T, const int *tri,        // [T][3]
    int* triangle_rects,          // [B * T][4]: h0, h_len, w0, w_len
    int *frag_counts              // [B * T]
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    const unsigned int b = idx / T;
    const unsigned int t = idx % T;

    const int t0 = tri[t * 3 + 0];
    const int t1 = tri[t * 3 + 1];
    const int t2 = tri[t * 3 + 2];

    const float fH = static_cast<float>(H);
    const float fW = static_cast<float>(W);
    float x0 = fW, y0 = fH;
    float x1 = 0.f, y1 = 0.f;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int v = (i == 0) ? t0 : (i == 1) ? t1 : t2;
        const float* p = &pos[(b * V + v) * 4];

        const float inv_w = 1.f / p[3];
        const float x = 0.5f * (1.f + p[0] * inv_w) * fW;
        const float y = 0.5f * (1.f - p[1] * inv_w) * fH;

        x0 = isnan(x) ? x0 : fminf(x, x0);
        x1 = isnan(x) ? x1 : fmaxf(x, x1);
        y0 = isnan(y) ? y0 : fminf(y, y0);
        y1 = isnan(y) ? y1 : fmaxf(y, y1);
    }

    const int h0 = MAX(static_cast<int>(floorf(y0)), 0);
    const int h1 = MIN(static_cast<int>(floorf(y1)), H - 1) + 1;
    const int w0 = MAX(static_cast<int>(floorf(x0)), 0);
    const int w1 = MIN(static_cast<int>(floorf(x1)), W - 1) + 1;

    const int h_len = MAX(h1 - h0, 0);
    const int w_len = MAX(w1 - w0, 0);

    triangle_rects[idx * 4 + 0] = h0;
    triangle_rects[idx * 4 + 1] = h_len;
    triangle_rects[idx * 4 + 2] = w0;
    triangle_rects[idx * 4 + 3] = w_len;
    frag_counts[idx] = h_len * w_len;
}

int compute_triangle_rects(
    int H, int W, int B,
    int V, const float* pos,     // [B * V][4]
    int T, const int* tri,       // [T][3]
    int* triangle_rects,         // [B * T][4]: h0, h_len, w0, w_len
    int* frag_prefix_sum         // [B * T]
) {
    const int total = B * T;
    int* frag_counts = nullptr;
    CUDA_CALL(cudaMalloc(&frag_counts, sizeof(int) * total));

    compute_triangle_rects_kernel<<<GET_BLOCKS(total), THREADS_PER_BLOCK>>>(
        H, W, B, V, pos, T, tri, triangle_rects, frag_counts
    );

    // Use Thrust for prefix sum on raw CUDA memory
    thrust::inclusive_scan(
        thrust::device_pointer_cast(frag_counts),
        thrust::device_pointer_cast(frag_counts + total),
        thrust::device_pointer_cast(frag_prefix_sum)
    );

    int num_frags = 0;
    CUDA_CALL(cudaMemcpy(&num_frags, frag_prefix_sum + (total - 1), sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(frag_counts));
    CUDA_CALL(cudaGetLastError());
    return num_frags;
}

__global__ void compute_fragments_kernel(
    int H, int W,
    int V, const float* pos,
    int T, const int* tri,
    int num_tris,  // == B * T
    int num_frags,
    const int* frag_prefix_sum,  // (num_tris,)
    const int* triangle_rects,   // (num_tris, 4)
    int* frag_pix,               // (num_frags, 3)
    float* frag_attrs            // (num_frags, 4)
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    // Binary search to find triangle index
    int lo = 0, hi = num_tris;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (frag_prefix_sum[mid] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    const int tri_idx = lo;
    const int b = tri_idx / T;
    const int t = tri_idx % T;

    const int* rect = &triangle_rects[tri_idx * 4];
    const int h0 = rect[0];
    const int h_len = rect[1];
    const int w0 = rect[2];
    const int w_len = rect[3];
    if (w_len <= 0 || h_len <= 0) return;

    const int local_idx = static_cast<int>(idx) - (tri_idx == 0 ? 0 : frag_prefix_sum[tri_idx - 1]);
    const int y = h0 + local_idx / w_len;
    const int x = w0 + local_idx % w_len;

    if (x < 0 || x >= W || y < 0 || y >= H) return;

    const float px = -1.f + 2.f * (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
    const float py = -1.f + 2.f * (static_cast<float>(y) + 0.5f) / static_cast<float>(H);

    const int i0 = tri[t * 3 + 0];
    const int i1 = tri[t * 3 + 1];
    const int i2 = tri[t * 3 + 2];
    const float* batch_pos = pos + b * V * 4;

    const float p0x = batch_pos[i0 * 4 + 0], p0y = batch_pos[i0 * 4 + 1], p0z = batch_pos[i0 * 4 + 2], p0w = batch_pos[i0 * 4 + 3];
    const float p1x = batch_pos[i1 * 4 + 0], p1y = batch_pos[i1 * 4 + 1], p1z = batch_pos[i1 * 4 + 2], p1w = batch_pos[i1 * 4 + 3];
    const float p2x = batch_pos[i2 * 4 + 0], p2y = batch_pos[i2 * 4 + 1], p2z = batch_pos[i2 * 4 + 2], p2w = batch_pos[i2 * 4 + 3];

    const float q0x = p0x - px * p0w, q0y = -p0y - py * p0w;
    const float q1x = p1x - px * p1w, q1y = -p1y - py * p1w;
    const float q2x = p2x - px * p2w, q2y = -p2y - py * p2w;

    const float A0 = q1x * q2y - q1y * q2x;
    const float A1 = q2x * q0y - q2y * q0x;
    const float A2 = q0x * q1y - q0y * q1x;
    const float A = A0 + A1 + A2;

    if (A == 0.f) return;
    const float invA = 1.f / A;
    const float b0 = A0 * invA;
    const float b1 = A1 * invA;

    const float z_clip = A0 * p0z + A1 * p1z + A2 * p2z;
    const float w_clip = A0 * p0w + A1 * p1w + A2 * p2w;
    const float z_ndc = z_clip / w_clip;

    if (!isfinite(b0) || !isfinite(b1) || !isfinite(z_ndc)) return;
    if (b0 < 0.f || b1 < 0.f || b0 + b1 > 1.f || z_ndc < -1.f || z_ndc > 1.f) return;

    frag_pix[idx * 3 + 0] = b;
    frag_pix[idx * 3 + 1] = y;
    frag_pix[idx * 3 + 2] = x;
    frag_attrs[idx * 4 + 0] = b0;
    frag_attrs[idx * 4 + 1] = b1;
    frag_attrs[idx * 4 + 2] = z_ndc;
    frag_attrs[idx * 4 + 3] = static_cast<float>(t + 1);
}

void compute_fragments(
    int H, int W,
    int V, const float* pos,
    int T, const int* tri,
    int num_tris,
    int num_frags,
    const int* frag_prefix_sum,
    const int* triangle_rects,
    int* frag_pix,
    float* frag_attrs
) {
    compute_fragments_kernel<<<GET_BLOCKS(num_frags), THREADS_PER_BLOCK>>>(
        H, W, V, pos, T, tri,
        num_tris,
        num_frags,
        frag_prefix_sum,
        triangle_rects,
        frag_pix,
        frag_attrs
    );
    CUDA_CALL(cudaGetLastError());
}

// --- Packing Helpers ---
__device__ __forceinline__ long long pack_depth_and_index(float zw, int index) {
    int zw_bits;
    memcpy(&zw_bits, &zw, sizeof(float)); // reinterpret float as int bits
    return (static_cast<long long>(zw_bits) << 32) | static_cast<unsigned int>(index);
}

__device__ __forceinline__ int unpack_index(long long packed) {
    return static_cast<int>(packed & 0xFFFFFFFFLL);
}

// --- Depth Test Kernel ---
__global__ void depth_test_kernel(
    int num_fragments,
    const int* __restrict__ frag_pix,        // (num_fragments, 3)
    const float* __restrict__ frag_attrs,    // (num_fragments, 4)
    int H, int W,
    long long* __restrict__ frag_idx         // (B * H * W), initialized to LLONG_MAX
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fragments) return;

    const int b = frag_pix[idx * 3 + 0];
    const int h = frag_pix[idx * 3 + 1];
    const int w = frag_pix[idx * 3 + 2];
    // if (b < 0 || h < 0 || w < 0) return;

    const float zw = frag_attrs[idx * 4 + 2];
    // if (zw <= -1.f || !isfinite(zw)) return;

    const long long packed = pack_depth_and_index(zw + 2.f, static_cast<int>(idx));
    atomicMin(&frag_idx[b * H * W + h * W + w], packed);
}

// --- Gathering Kernel ---
__global__ void gather_depth_test_kernel(
    int B, int H, int W,
    const long long* __restrict__ frag_idx,
    const float* __restrict__ frag_attrs,
    float* __restrict__ rast_out             // (B, H, W, 4)
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * W;
    if (idx >= total) return;

    const long long packed = frag_idx[idx];
    const int frag_i = unpack_index(packed);

    if (frag_i < 0) return;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        rast_out[idx * 4 + k] = frag_attrs[frag_i * 4 + k];
    }
}

__global__ void fill_ll_max(long long* arr, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] = LLONG_MAX;
}

void depth_test(
    int B, int H, int W,
    int num_fragments,
    const int* frag_pix,   // (num_fragments, 3)
    const float* frag_attrs,   // (num_fragments, 4)
    float* rast_out            // (B, H, W, 4)
) {
    const int total = B * H * W;

    long long* frag_idx = nullptr;
    CUDA_CALL(cudaMalloc(&frag_idx, sizeof(long long) * total));
    fill_ll_max<<<GET_BLOCKS(total), THREADS_PER_BLOCK>>>(frag_idx, total);

    depth_test_kernel<<<GET_BLOCKS(num_fragments), THREADS_PER_BLOCK>>>(
        num_fragments, frag_pix, frag_attrs, H, W, frag_idx);

    gather_depth_test_kernel<<<GET_BLOCKS(total), THREADS_PER_BLOCK>>>(
        B, H, W, frag_idx, frag_attrs, rast_out);

    CUDA_CALL(cudaFree(frag_idx));
}

__global__ void filter_valid_fragments_kernel(
    int num_frags,
    const int* __restrict__ frag_pix,        // [num_frags, 3]
    const float* __restrict__ frag_attrs,    // [num_frags, 4]
    int* __restrict__ frag_pix_out,          // [num_frags, 3] (preallocated)
    float* __restrict__ frag_attrs_out,      // [num_frags, 4]
    int* __restrict__ global_counter         // [1]
) {
    extern __shared__ int shared_scan[]; // shared memory for scan and block count
    int* valid_flags = shared_scan;
    int* block_base  = shared_scan + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Step 1: Each thread checks if valid
    int is_valid = 0;
    if (idx < num_frags) {
        int frag_val = frag_pix[3 * idx + 0];
        is_valid = (frag_val >= 0);
    }
    valid_flags[tid] = is_valid;
    __syncthreads();

    // Step 2: Inclusive scan (Hillis-Steele)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset)
            temp = valid_flags[tid - offset];
        __syncthreads();
        valid_flags[tid] += temp;
        __syncthreads();
    }

    // Step 3: First thread in block gets total count
    if (tid == blockDim.x - 1) {
        int total = valid_flags[tid];
        block_base[0] = atomicAdd(global_counter, total);
    }
    __syncthreads();

    // Step 4: Write to output if valid
    if (idx < num_frags && is_valid) {
        int local_idx = (tid > 0) ? valid_flags[tid - 1] : 0;
        int output_idx = block_base[0] + local_idx;

        // Copy frag_pix (3 ints)
        for (int i = 0; i < 3; ++i)
            frag_pix_out[3 * output_idx + i] = frag_pix[3 * idx + i];

        // Copy frag_attrs (4 floats)
        for (int i = 0; i < 4; ++i)
            frag_attrs_out[4 * output_idx + i] = frag_attrs[4 * idx + i];
    }
}

int filter_valid_fragments(
    int num_frags,
    const int* frag_pix,         // [num_frags, 3]
    const float* frag_attrs,     // [num_frags, 4]
    int* frag_pix_out,           // [num_frags, 3]
    float* frag_attrs_out        // [num_frags, 4]
) {
    int valid_count = 0;
    if (num_frags == 0) return valid_count;

    int* global_counter = nullptr;
    CUDA_CALL(cudaMalloc(&global_counter, sizeof(int)));
    CUDA_CALL(cudaMemset(global_counter, 0, sizeof(int)));

    const size_t shared_mem = sizeof(int) * (THREADS_PER_BLOCK + 1);

    filter_valid_fragments_kernel<<<GET_BLOCKS(num_frags), THREADS_PER_BLOCK, shared_mem>>>(
        num_frags,
        frag_pix,
        frag_attrs,
        frag_pix_out,
        frag_attrs_out,
        global_counter
    );

    CUDA_CALL(cudaMemcpy(&valid_count, global_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(global_counter));
    CUDA_CALL(cudaGetLastError()); // catch kernel errors
    return valid_count;
}

__global__ void interpolate_triangle_attributes_kernel(
    int B, int H, int W,
    const float *rast,                // [B, H, W, 4]
    int attr_dim,                     // C
    const float *attr,                // [V, C]
    const int *tri,                   // [T, 3]
    float *image                      // [B, H, W, C]
) {
    const unsigned int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= B * H * W) return;

    const unsigned int rast_offset = pixel_idx * 4;
    const unsigned int image_offset = pixel_idx * attr_dim;

    const float bary0 = rast[rast_offset + 0];
    const float bary1 = rast[rast_offset + 1];
    const float bary2 = 1.f - bary0 - bary1;
    const int triangle_index = static_cast<int>(rast[rast_offset + 3]) - 1;

    if (triangle_index < 0) return;

    const int v0 = tri[triangle_index * 3 + 0];
    const int v1 = tri[triangle_index * 3 + 1];
    const int v2 = tri[triangle_index * 3 + 2];

    const float *attr0 = &attr[attr_dim * v0];
    const float *attr1 = &attr[attr_dim * v1];
    const float *attr2 = &attr[attr_dim * v2];

#pragma unroll 8
    for (int d = 0; d < attr_dim; ++d) {
        image[image_offset + d] = bary0 * attr0[d] + bary1 * attr1[d] + bary2 * attr2[d];
    }
}

__global__ void backward_interpolate_triangle_attributes_kernel(
    int B, int H, int W,
    const float *rast,                // [B, H, W, 4]
    int attr_dim,                     // C
    const float *d_image,             // [B, H, W, C]
    const int *tri,                   // [T, 3]
    float *d_attr                     // [V, C]
) {
    const unsigned int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= B * H * W) return;

    const unsigned int rast_offset = pixel_idx * 4;
    const unsigned int image_offset = pixel_idx * attr_dim;

    const float bary0 = rast[rast_offset + 0];
    const float bary1 = rast[rast_offset + 1];
    const float bary2 = 1.f - bary0 - bary1;
    const int triangle_index = static_cast<int>(rast[rast_offset + 3]) - 1;

    if (triangle_index < 0) return;

    const int v0 = tri[triangle_index * 3 + 0];
    const int v1 = tri[triangle_index * 3 + 1];
    const int v2 = tri[triangle_index * 3 + 2];

    float *grad0 = &d_attr[attr_dim * v0];
    float *grad1 = &d_attr[attr_dim * v1];
    float *grad2 = &d_attr[attr_dim * v2];

#pragma unroll 8
    for (int d = 0; d < attr_dim; ++d) {
        const float d_val = d_image[image_offset + d];
        atomicAdd(&grad0[d], d_val * bary0);
        atomicAdd(&grad1[d], d_val * bary1);
        atomicAdd(&grad2[d], d_val * bary2);
    }
}

void interpolate_triangle_attributes(
    int B, int H, int W,
    int attr_dim,
    const float* rast,     // [B, H, W, 4]
    const float* attr,     // [V, attr_dim]
    const int* tri,        // [T, 3]
    float* image           // [B, H, W, attr_dim]
) {
    const int total_pixels = B * H * W;
    interpolate_triangle_attributes_kernel<<<GET_BLOCKS(total_pixels), THREADS_PER_BLOCK>>>(
        B, H, W, rast,
        attr_dim, attr, tri,
        image
    );
    CUDA_CALL(cudaGetLastError());
}

void backward_interpolate_triangle_attributes(
    int B, int H, int W,
    int attr_dim,
    const float* rast,      // [B, H, W, 4]
    const float* d_image,   // [B, H, W, attr_dim]
    const int* tri,         // [T, 3]
    float* d_attr           // [V, attr_dim]
) {
    const int total_pixels = B * H * W;
    backward_interpolate_triangle_attributes_kernel<<<GET_BLOCKS(total_pixels), THREADS_PER_BLOCK>>>(
        B, H, W, rast,
        attr_dim, d_image, tri,
        d_attr
    );
    CUDA_CALL(cudaGetLastError());
}

__global__ void cluster_mask_from_fragments_kernel(
    int num_frags,
    const int* frag_pix,        // [num_frags, 3]: (batch, y, x)
    const float* frag_attrs,    // [num_frags, 4]: (..., ..., ..., prim_id + 1)
    const int* prim2cluster,    // [num_prims]: primitive index → cluster ID
    int H, int W, int num_slots,
    int* bitset                 // [B * H * W * num_slots]: output mask
) {
    const unsigned int frag_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frag_idx >= num_frags) return;

    const int batch_idx = frag_pix[frag_idx * 3 + 0];
    const int y = frag_pix[frag_idx * 3 + 1];
    const int x = frag_pix[frag_idx * 3 + 2];

    const int prim_idx = static_cast<int>(frag_attrs[frag_idx * 4 + 3]) - 1;
    if (prim_idx < 0) return;

    const int cluster_id = prim2cluster[prim_idx];
    const int slot_idx = cluster_id / 32;
    const int bit_idx  = cluster_id % 32;

    const int pixel_idx = ((batch_idx * H + y) * W + x) * num_slots + slot_idx;
    atomicOr(&bitset[pixel_idx], 1 << bit_idx);
}

void cluster_mask_from_fragments(
    int num_frags,
    const int* frag_pix,        // [num_frags, 3]
    const float* frag_attrs,    // [num_frags, 4]
    const int* prim2cluster,    // [num_prims]
    int H, int W,
    int num_slots,
    int* bitset                 // [B * H * W * num_slots]
) {
    if (num_frags == 0) return;
    cluster_mask_from_fragments_kernel<<<GET_BLOCKS(num_frags), THREADS_PER_BLOCK>>>(
        num_frags,
        frag_pix,
        frag_attrs,
        prim2cluster,
        H, W,
        num_slots,
        bitset
    );
    CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace strns2cards
