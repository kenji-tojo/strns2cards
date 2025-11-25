#include "cuda_common.cuh"

namespace strns2cards {
namespace cuda {

// --- Helper: Find third vertex of triangle given two edge vertices
__device__ __forceinline__
int complete_triangle(const int *tri, int tri_idx, int v0, int v1) {
    const int i0 = tri[tri_idx * 3 + 0];
    const int i1 = tri[tri_idx * 3 + 1];
    const int i2 = tri[tri_idx * 3 + 2];
    return (i0 != v0 && i0 != v1) ? i0 :
           (i1 != v0 && i1 != v1) ? i1 :
           (i2 != v0 && i2 != v1) ? i2 : -1;
}

// --- Helper: NDC to pixel-space x/y
__device__ __forceinline__
float ndc_to_pixel_x(float ndc_x, float fW) {
    return fW * 0.5f * (1.f + ndc_x);
}
__device__ __forceinline__
float ndc_to_pixel_y(float ndc_y, float fH) {
    return fH * 0.5f * (1.f - ndc_y);
}

// --- Helper: Project homogeneous vertex to pixel
__device__ __forceinline__
void project_vertex(const float* pos, int idx, float fW, float fH, float& x, float& y) {
    const float w = pos[idx * 4 + 3];
    const float ndc_x = pos[idx * 4 + 0] / w;
    const float ndc_y = pos[idx * 4 + 1] / w;
    x = ndc_to_pixel_x(ndc_x, fW);
    y = ndc_to_pixel_y(ndc_y, fH);
}

// --- Edge marking kernel (detect boundary/silhouette edges)
__global__ void mark_discontinuity_edges_kernel(
    int H, int W, int B,
    int V, const float *pos, const int *tri, int E, const int *edges, const int *edge2tri,
    int *prim_ids, float *normals
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * E) return;

    const unsigned int batch_idx = idx / E;
    const unsigned int edge_idx  = idx % E;

    const float* pos_batch = pos + batch_idx * V * 4;
    const float fH = static_cast<float>(H);
    const float fW = static_cast<float>(W);

    const int v0 = edges[edge_idx * 2 + 0];
    const int v1 = edges[edge_idx * 2 + 1];

    float x0, y0, x1, y1;
    project_vertex(pos_batch, v0, fW, fH, x0, y0);
    project_vertex(pos_batch, v1, fW, fH, x1, y1);

    const float edge_dx = x1 - x0;
    const float edge_dy = y1 - y0;
    const float edge_len = sqrtf(edge_dx * edge_dx + edge_dy * edge_dy);
    if (edge_len == 0.f) return;

    const float nx = -edge_dy / edge_len;
    const float ny =  edge_dx / edge_len;

    float dot_tri0 = 0.f, dot_tri1 = 0.f;

    const int tri0 = edge2tri[edge_idx * 2 + 0];
    const int tri1 = edge2tri[edge_idx * 2 + 1];

    if (tri0 >= 0) {
        const int v2 = complete_triangle(tri, tri0, v0, v1);
        float x2, y2;
        project_vertex(pos_batch, v2, fW, fH, x2, y2);
        dot_tri0 = nx * (x2 - x0) + ny * (y2 - y0);
    }

    if (tri1 >= 0) {
        const int v2 = complete_triangle(tri, tri1, v0, v1);
        float x2, y2;
        project_vertex(pos_batch, v2, fW, fH, x2, y2);
        dot_tri1 = nx * (x2 - x0) + ny * (y2 - y0);
    }

    if (dot_tri0 * dot_tri1 >= 0.f) {
        prim_ids[idx] = static_cast<int>(idx);
    }

    const float sign = 2.f * (.5f - static_cast<float>(dot_tri0 > 0.f || dot_tri1 > 0.f));
    normals[idx * 2 + 0] = sign * nx;
    normals[idx * 2 + 1] = sign * ny;
}

// --- Launcher for edge discontinuity marking (prepass)
void mark_discontinuity_edges(
    int H, int W, int B,
    int V, const float* pos,
    const int* tri,
    int E, const int* edges,
    const int* edge2tri,
    int* prim_ids,
    float* normals
) {
    const int total_threads = B * E;
    if (total_threads == 0) return;

    mark_discontinuity_edges_kernel<<<GET_BLOCKS(total_threads), THREADS_PER_BLOCK>>>(
        H, W, B,
        V, pos,
        tri,
        E, edges,
        edge2tri,
        prim_ids,
        normals
    );
    CUDA_CALL(cudaGetLastError());
}

// --- Compute Gaussian weight at (w, h) relative to continuous sample (x, y)
__device__ __forceinline__
float gaussian_weight(float x, float y, int w, int h, float R = 2.f) {
    constexpr float pi = 3.14159265358979323846f;
    const float sigma = R / 4.f;
    const float c = 1.f / (sigma * sigma * 2.f * pi);
    const float alpha = 0.5f / (sigma * sigma);
    const float dx = static_cast<float>(w) + 0.5f - x;
    const float dy = static_cast<float>(h) + 0.5f - y;
    const float norm2 = dx * dx + dy * dy;
    return c * __expf(-alpha * norm2) * static_cast<float>(norm2 < R * R);
}

// --- Read bit color from cluster bitset
__device__ __forceinline__
float read_cluster_bit_color(
    int batch_idx, int w, int h, int cluster_id,
    int H, int W, int num_slots, const int* bitset
) {
    const int pix_idx = H * W * batch_idx + w + h * W;
    const int slot = cluster_id / 32;
    const int bit = cluster_id % 32;
    return static_cast<float>((bitset[pix_idx * num_slots + slot] >> bit) & 1);
}

// --- Adjoint sum (weighted error around pixel for cluster)
__device__ __forceinline__
float compute_adjoint_sum(
    int batch_idx, float x, float y, int cluster_id,
    int H, int W, int num_slots,
    const int* bitset, const int* target_bitset,
    float kernel_radius, float rho
) {
    const int R = static_cast<int>(floorf(kernel_radius));
    const int center_w = static_cast<int>(floorf(x));
    const int center_h = static_cast<int>(floorf(y));
    const int h0 = max(center_h - R, 0);
    const int h1 = min(center_h + R, H - 1) + 1;
    const int w0 = max(center_w - R, 0);
    const int w1 = min(center_w + R, W - 1) + 1;

    const int slot = cluster_id / 32;
    const int bit = cluster_id % 32;
    float dc = 0.f;

    for (int h = h0; h < h1; ++h) {
        for (int w = w0; w < w1; ++w) {
            const int pix_idx = H * W * batch_idx + w + h * W;
            const float c0 = static_cast<float>((bitset[pix_idx * num_slots + slot] >> bit) & 1);
            const float c1 = static_cast<float>((target_bitset[pix_idx * num_slots + slot] >> bit) & 1);
            const float weight = gaussian_weight(x, y, w, h, kernel_radius);
            dc += weight * (c0 - c1) * (c1 == 0.f ? rho : 1.f);
        }
    }
    return dc;
}

// --- Backward kernel for anti-aliased cluster bitset optimization
__global__ void backward_antialiased_cluster_bitset_kernel(
    int num_frags,
    const int* frag_pix,
    const float* frag_attrs_dda,
    int H, int W,
    int num_slots,
    const int* bitset,
    const int* target_bitset,
    int V, const float* pos,
    int E, const int* edges,
    const float* normals,
    const int* edge2cluster,
    float* d_pos,
    float kernel_radius,
    float rho
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frags) return;

    const int batch_idx = frag_pix[idx * 3 + 0];
    const int h = frag_pix[idx * 3 + 1];
    const int w = frag_pix[idx * 3 + 2];

    const float axis = frag_attrs_dda[idx * 4 + 0];
    const float t1   = frag_attrs_dda[idx * 4 + 1];
    const float t0   = 1.f - t1;
    const int edge_idx = static_cast<int>(frag_attrs_dda[idx * 4 + 3]) - 1;
    if (edge_idx < 0) return;

    const float* pos_batch = pos + batch_idx * V * 4;
    const float* normal_batch = normals + batch_idx * E * 2;
    float* d_pos_batch = d_pos + batch_idx * V * 4;

    const float fH = static_cast<float>(H);
    const float fW = static_cast<float>(W);

    const int v0 = edges[edge_idx * 2 + 0];
    const int v1 = edges[edge_idx * 2 + 1];

    const float p0x = pos_batch[v0 * 4 + 0];
    const float p0y = pos_batch[v0 * 4 + 1];
    const float p0w = pos_batch[v0 * 4 + 3];
    const float p1x = pos_batch[v1 * 4 + 0];
    const float p1y = pos_batch[v1 * 4 + 1];
    const float p1w = pos_batch[v1 * 4 + 3];

    const float px = t0 * p0x + t1 * p1x;
    const float py = t0 * p0y + t1 * p1y;
    const float pw = t0 * p0w + t1 * p1w;

    const float fx = 0.5f * (1.f + px / pw) * fW;
    const float fy = 0.5f * (1.f - py / pw) * fH;

    const float nx = normal_batch[edge_idx * 2 + 0];
    const float ny = normal_batch[edge_idx * 2 + 1];

    int neighbor_h = h, neighbor_w = w;
    const float fx_rel = fx - floorf(fx);
    const float fy_rel = fy - floorf(fy);

    constexpr float eps = 1e-4f;
    float dot = 0.f;

    if (axis == 0.f) {
        if (!isfinite(fx_rel) || fabsf(fx_rel - 0.5f) >= eps) return;
        int d = 2 * static_cast<int>(fy_rel > 0.5f) - 1;
        neighbor_h += d;
        dot += static_cast<float>(d) * ny;
    } else {
        if (!isfinite(fy_rel) || fabsf(fy_rel - 0.5f) >= eps) return;
        int d = 2 * static_cast<int>(fx_rel > 0.5f) - 1;
        neighbor_w += d;
        dot += static_cast<float>(d) * nx;
    }

    if (neighbor_w < 0 || neighbor_w >= W || neighbor_h < 0 || neighbor_h >= H || dot == 0.f) return;

    const int cluster_id = edge2cluster[edge_idx];
    const float c01 = (dot > 0.f) ?
        1.f - read_cluster_bit_color(batch_idx, neighbor_w, neighbor_h, cluster_id, H, W, num_slots, bitset) :
        1.f - read_cluster_bit_color(batch_idx, w, h, cluster_id, H, W, num_slots, bitset);

    const float dc = c01 * compute_adjoint_sum(batch_idx, fx, fy, cluster_id, H, W, num_slots, bitset, target_bitset, kernel_radius, rho);
    if (dc == 0.f) return;

    const float dx = dc * 0.5f * fW * nx;
    const float dy = dc * -0.5f * fH * ny;

    const float dpx = dx / pw;
    const float dpy = dy / pw;
    const float dpw = (dx * px + dy * py) / (pw * pw);

    if (isfinite(dpx) && isfinite(dpy) && isfinite(dpw)) {
        atomicAdd(&d_pos_batch[v0 * 4 + 0], dpx * t0);
        atomicAdd(&d_pos_batch[v0 * 4 + 1], dpy * t0);
        atomicAdd(&d_pos_batch[v0 * 4 + 3], dpw * t0);
        atomicAdd(&d_pos_batch[v1 * 4 + 0], dpx * t1);
        atomicAdd(&d_pos_batch[v1 * 4 + 1], dpy * t1);
        atomicAdd(&d_pos_batch[v1 * 4 + 3], dpw * t1);
    }
}

// --- Launcher for backward anti-aliased cluster bitset optimization
void backward_antialiased_cluster_bitset(
    int num_frags,
    const int* frag_pix,
    const float* frag_attrs_dda,
    int H, int W,
    int num_slots,
    const int* bitset,
    const int* target_bitset,
    int V, const float* pos,
    int E, const int* edges,
    const float* normals,
    const int* edge2cluster,
    float* d_pos,
    float kernel_radius,
    float rho
) {
    if (num_frags == 0) return;

    backward_antialiased_cluster_bitset_kernel<<<GET_BLOCKS(num_frags), THREADS_PER_BLOCK>>>(
        num_frags,
        frag_pix,
        frag_attrs_dda,
        H, W,
        num_slots,
        bitset,
        target_bitset,
        V, pos,
        E, edges,
        normals,
        edge2cluster,
        d_pos,
        kernel_radius,
        rho
    );
    CUDA_CALL(cudaGetLastError());
}

} // namespace cuda
} // namespace strns2cards
