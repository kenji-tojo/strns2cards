#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "hair_loader.h"
#include "mesh_utils.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace strns2cards {
namespace cuda {

int compute_triangle_rects(
    int H, int W, int B,
    int V, const float* pos,     // [B * V][4]
    int T, const int* tri,       // [T][3]
    int* triangle_rects,         // [B * T][4]: h0, h_len, w0, w_len
    int* frag_prefix_sum         // [B * T]
);

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
);

void depth_test(
    int B, int H, int W,
    int num_fragments,
    const int* frag_pix,   // (num_fragments, 3)
    const float* frag_attrs,   // (num_fragments, 4)
    float* rast_out            // (B, H, W, 4)
);

int filter_valid_fragments(
    int num_frags,
    const int* frag_pix,         // [num_frags, 3]
    const float* frag_attrs,     // [num_frags, 4]
    int* frag_pix_out,           // [num_frags, 3]
    float* frag_attrs_out        // [num_frags, 4]
);

void interpolate_triangle_attributes(
    int B, int H, int W,
    int attr_dim,
    const float* rast,     // [B, H, W, 4]
    const float* attr,     // [V, attr_dim]
    const int* tri,        // [T, 3]
    float* image           // [B, H, W, attr_dim]
);

void backward_interpolate_triangle_attributes(
    int B, int H, int W,
    int attr_dim,
    const float* rast,      // [B, H, W, 4]
    const float* d_image,   // [B, H, W, attr_dim]
    const int* tri,         // [T, 3]
    float* d_attr           // [V, attr_dim]
);

void cluster_mask_from_fragments(
    int num_frags,
    const int* frag_pix,        // [num_frags, 3]
    const float* frag_attrs,    // [num_frags, 4]
    const int* prim2cluster,    // [num_prims]
    int H, int W,
    int num_slots,
    int* bitset                 // [B * H * W * num_slots]
);

void accumulate_bitset_rgb(
    int B, int H, int W, int num_slots,
    const int* bitset,              // [B * H * W * num_slots]
    const float* cluster_rgb,       // [num_clusters, 3]
    float* accum_color              // [B * H * W * 3]
);

void popcount_bitset(
    int B, int H, int W, int num_slots,
    const int* bitset,     // [B * H * W * num_slots]
    float* count_image     // [B * H * W]
);

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
);

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
);

void dda_interpolate_attributes(
    int B, int H, int W, int C,
    const float* rast_dda,    // [B * H * W, 4]
    const float* attr,        // [V, C]
    const int* edges,         // [E, 2]
    float* image              // [B * H * W, C]
);

void backward_dda_interpolate_attributes(
    int B, int H, int W,
    const float* rast_dda,
    int C,
    const float* d_image,
    const int* edges,
    float* d_attr
);

void mark_discontinuity_edges(
    int H, int W, int B,
    int V, const float* pos,
    const int* tri,
    int E, const int* edges,
    const int* edge2tri,
    int* prim_ids,
    float* normals
);

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
);

int compact_valid_ints(
    int N,
    const int* input,     // [N]
    int* output           // [N] (preallocated)
);

void smooth_strands(
    int num_strands,
    int num_points,
    const float* strands,    // (N, V, 3)
    const float* arclength,  // (N, V, 3)
    float* smoothed_strands,
    float beta
);

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
);

} // namespace cuda
} // namespace strns2cards

namespace nb = nanobind;

namespace {
// Helper for C++14: Check if a string ends with a given suffix
bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}
} // namespace

NB_MODULE(_core, m) {

    m.def("print_version", [] () { printf("pystrns2cards version 0.1.0\n"); });

    m.def("build_edges_from_triangles", [](
        nb::ndarray<int32_t, nb::numpy, nb::shape<-1, 3>, nb::c_contig> tri
    ) -> auto {
        const int T = static_cast<int>(tri.shape(0));
        const int32_t* tri_ptr = tri.data();

        std::vector<int> edges_flat, edge2tri_flat;
        strns2cards::build_edges_from_triangles(T, tri_ptr, edges_flat, edge2tri_flat);

        if (edges_flat.size() != edge2tri_flat.size()) {
            throw std::runtime_error("build_edges_from_triangles: mismatch in generated edge array sizes.");
        }

        // Allocate raw memory using malloc and copy
        const size_t num_values = edges_flat.size();  // = E * 2
        const size_t num_edges = num_values / 2;

        auto* edges_ptr = new int32_t[num_values];
        auto* edge2tri_ptr = new int32_t[num_values];

        std::memcpy(edges_ptr, edges_flat.data(), sizeof(int32_t) * num_values);
        std::memcpy(edge2tri_ptr, edge2tri_flat.data(), sizeof(int32_t) * num_values);

        // Each edge is 2 ints → shape (E, 2)
        auto edges_array = nb::ndarray<int32_t, nb::numpy, nb::shape<-1, 2>, nb::c_contig>(
            edges_ptr, {num_edges, 2}, {}
        );

        // Each edge2tri is 2 ints → shape (E, 2)
        auto edge2tri_array = nb::ndarray<int32_t, nb::numpy, nb::shape<-1, 2>, nb::c_contig>(
            edge2tri_ptr, {num_edges, 2}, {}
        );

        return nb::make_tuple(edges_array, edge2tri_array);
    }, "Build unique edge list and triangle mapping from face indices",
    nb::rv_policy::take_ownership);

    m.def("compute_triangle_rects", [](
        int H, int W,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,               // [B, V, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,     nb::c_contig> tri,               // [T, 3]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 4>,     nb::c_contig> triangle_rects,    // [B * T, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1>,        nb::c_contig> frag_prefix_sum    // [B * T]
    ) -> int {
        if (pos.device_id() < 0 || tri.device_id() < 0 ||
            triangle_rects.device_id() < 0 || frag_prefix_sum.device_id() < 0) {
            throw std::runtime_error("All tensors must be CUDA tensors (device_id >= 0).");
        }

        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int T = static_cast<int>(tri.shape(0));

        if (triangle_rects.shape(0) != B *  T) {
            throw std::runtime_error("`triangle_rects` must have shape [B * T, 4].");
        }

        if (frag_prefix_sum.shape(0) != B * T) {
            throw std::runtime_error("`frag_prefix_sum` must have shape [B * T].");
        }

        const float* pos_ptr = pos.data();
        const int32_t* tri_ptr = tri.data();
        int32_t* triangle_rects_ptr = triangle_rects.data();
        int32_t* frag_prefix_sum_ptr = frag_prefix_sum.data();

        return strns2cards::cuda::compute_triangle_rects(
            H, W, B,
            V, pos_ptr,
            T, tri_ptr,
            triangle_rects_ptr,
            frag_prefix_sum_ptr
        );
    }, "Compute screen-space bounding rectangles and fragment prefix sums for each triangle");

    m.def("compute_fragments", [](
        int H, int W,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,     nb::c_contig> tri,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1>,        nb::c_contig> frag_prefix_sum,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 4>,     nb::c_contig> triangle_rects,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,     nb::c_contig> frag_pix,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>,     nb::c_contig> frag_attrs
    ) {
        if (pos.device_id() < 0 || tri.device_id() < 0 ||
            frag_prefix_sum.device_id() < 0 || triangle_rects.device_id() < 0 ||
            frag_pix.device_id() < 0 || frag_attrs.device_id() < 0) {
            throw std::runtime_error("All tensors must be CUDA tensors (device_id >= 0).");
        }

        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int T = static_cast<int>(tri.shape(0));
        const int num_tris = B * T;
        const int num_fragments = static_cast<int>(frag_pix.shape(0));

        if (frag_attrs.shape(0) != num_fragments || frag_prefix_sum.shape(0) != num_tris) {
            throw std::runtime_error("Mismatch in fragment count or prefix sum length.");
        }

        const float* pos_ptr = pos.data();
        const int32_t* tri_ptr = tri.data();
        const int32_t* frag_prefix_sum_ptr = frag_prefix_sum.data();
        const int32_t* triangle_rects_ptr = triangle_rects.data();
        int32_t* frag_pix_ptr = frag_pix.data();
        float* frag_attrs_ptr = frag_attrs.data();

        strns2cards::cuda::compute_fragments(
            H, W,
            V, pos_ptr,
            T, tri_ptr,
            num_tris,
            num_fragments,
            frag_prefix_sum_ptr,
            triangle_rects_ptr,
            frag_pix_ptr,
            frag_attrs_ptr
        );
    }, "Compute rasterization fragments with barycentric coordinates and triangle ID");

    m.def("depth_test", [](
        int H, int W,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast_out
    ) {
        if (frag_pix.device_id() < 0 || frag_attrs.device_id() < 0 || rast_out.device_id() < 0) {
            throw std::runtime_error("All tensors must be on a CUDA device (device_id >= 0)");
        }

        const int num_fragments = static_cast<int>(frag_pix.shape(0));
        const int B = static_cast<int>(rast_out.shape(0));

        if (frag_attrs.shape(0) != num_fragments) {
            throw std::runtime_error("frag_attrs must match frag_pix in size.");
        }
        if (rast_out.shape(1) != H || rast_out.shape(2) != W) {
            throw std::runtime_error("rast_out must have shape (B, H, W, 4).");
        }

        const int32_t* frag_pix_ptr = frag_pix.data();
        const float* frag_attrs_ptr = frag_attrs.data();
        float* rast_out_ptr = rast_out.data();

        strns2cards::cuda::depth_test(
            B, H, W, num_fragments,
            frag_pix_ptr, frag_attrs_ptr, rast_out_ptr
        );
    }, "Rasterize visible fragments via z-buffer depth test");

    m.def("filter_valid_fragments", [](
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs,
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix_out,
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs_out
    ) -> int {
        // --- Device check (device_id < 0 means CPU) ---
        if (frag_pix.device_id() < 0 || frag_attrs.device_id() < 0 ||
            frag_pix_out.device_id() < 0 || frag_attrs_out.device_id() < 0) {
            throw std::runtime_error("All tensors must be CUDA tensors (device_id >= 0).");
        }

        // --- Shape check ---
        const int num_frags = static_cast<int>(frag_pix.shape(0));
        if (frag_attrs.shape(0) != num_frags ||
            frag_pix_out.shape(0) != num_frags ||
            frag_attrs_out.shape(0) != num_frags) {
            throw std::runtime_error("Mismatch in fragment counts between input/output tensors.");
        }

        // --- Data pointers ---
        const int* frag_pix_ptr       = frag_pix.data();
        const float* frag_attrs_ptr   = frag_attrs.data();
        int* frag_pix_out_ptr         = frag_pix_out.data();
        float* frag_attrs_out_ptr     = frag_attrs_out.data();

        // --- Launch CUDA ---
        return strns2cards::cuda::filter_valid_fragments(
            num_frags,
            frag_pix_ptr,
            frag_attrs_ptr,
            frag_pix_out_ptr,
            frag_attrs_out_ptr
        );
    }, "Filter valid fragments where frag_pix[:, 0] >= 0 using CUDA");

    m.def("interpolate_triangle_attributes", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,     // [B, H, W, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1>,         nb::c_contig> attr,     // [V, C]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,          nb::c_contig> tri,      // [T, 3]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> image     // [B, H, W, C]
    ) {
        if (rast.device_id() < 0 || attr.device_id() < 0 || tri.device_id() < 0 || image.device_id() < 0)
            throw std::runtime_error("All tensors must be on a CUDA device");

        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));
        const int attr_dim = static_cast<int>(attr.shape(1));

        strns2cards::cuda::interpolate_triangle_attributes(
            B, H, W, attr_dim,
            rast.data(),
            attr.data(),
            tri.data(),
            image.data()
        );
    }, "Forward interpolation of per-vertex attributes to pixels using barycentric coords");

    m.def("backward_interpolate_triangle_attributes", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast,     // [B, H, W, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> d_image,  // [B, H, W, C]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,          nb::c_contig> tri,      // [T, 3]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1>,         nb::c_contig> d_attr    // [V, C]
    ) {
        if (rast.device_id() < 0 || d_image.device_id() < 0 || tri.device_id() < 0 || d_attr.device_id() < 0)
            throw std::runtime_error("All tensors must be on a CUDA device");

        const int B = static_cast<int>(rast.shape(0));
        const int H = static_cast<int>(rast.shape(1));
        const int W = static_cast<int>(rast.shape(2));
        const int attr_dim = static_cast<int>(d_attr.shape(1));

        strns2cards::cuda::backward_interpolate_triangle_attributes(
            B, H, W, attr_dim,
            rast.data(),
            d_image.data(),
            tri.data(),
            d_attr.data()
        );
    }, "Backward pass for interpolating per-pixel gradients to vertex attributes");

    m.def("cluster_mask_from_fragments", [](
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 3>,       nb::c_contig> frag_pix,       // [N, 3]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, 4>,       nb::c_contig> frag_attrs,     // [N, 4]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1>,          nb::c_contig> prim2cluster,   // [num_prims]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> bitset       // [B, H, W, num_slots]
    ) {
        // --- Device check ---
        if (frag_pix.device_id() < 0 || frag_attrs.device_id() < 0 ||
            prim2cluster.device_id() < 0 || bitset.device_id() < 0) {
            throw std::runtime_error("All tensors must be CUDA tensors (device_id >= 0).");
        }

        const int num_frags = static_cast<int>(frag_pix.shape(0));
        if (frag_attrs.shape(0) != num_frags) {
            throw std::runtime_error("frag_pix and frag_attrs must have the same number of fragments.");
        }

        // --- Infer dimensions ---
        const int B = static_cast<int>(bitset.shape(0));
        const int H = static_cast<int>(bitset.shape(1));
        const int W = static_cast<int>(bitset.shape(2));
        const int num_slots = static_cast<int>(bitset.shape(3));

        // --- Pointers ---
        const int* frag_pix_ptr     = frag_pix.data();
        const float* frag_attrs_ptr = frag_attrs.data();
        const int* prim2cluster_ptr = prim2cluster.data();
        int* bitset_ptr             = bitset.data();

        // --- Launch CUDA ---
        strns2cards::cuda::cluster_mask_from_fragments(
            num_frags,
            frag_pix_ptr,
            frag_attrs_ptr,
            prim2cluster_ptr,
            H, W, num_slots,
            bitset_ptr
        );
    }, "Rasterize per-pixel cluster bitmask from general primitives (e.g., triangles or lines)");

    m.def("accumulate_bitset_rgb", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> bitset,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 3>, nb::c_contig> cluster_rgb,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1, 3>, nb::c_contig> accum_color
    ) {
        if (bitset.device_id() < 0 || cluster_rgb.device_id() < 0 || accum_color.device_id() < 0) {
            throw std::runtime_error("All tensors must be on CUDA");
        }

        const int B = static_cast<int>(bitset.shape(0));
        const int H = static_cast<int>(bitset.shape(1));
        const int W = static_cast<int>(bitset.shape(2));
        const int num_slots = static_cast<int>(bitset.shape(3));

        const int num_clusters = static_cast<int>(cluster_rgb.shape(0));
        if (num_slots * 32 < num_clusters) {
            throw std::runtime_error("Too few slots for the number of clusters.");
        }

        strns2cards::cuda::accumulate_bitset_rgb(
            B, H, W, num_slots,
            bitset.data(),
            cluster_rgb.data(),
            accum_color.data()
        );
    }, "Accumulate RGB color from bitset image using cluster RGB mapping");

    m.def("popcount_bitset", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> bitset,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, -1>, nb::c_contig> count_image
    ) {
        if (bitset.device_id() < 0 || count_image.device_id() < 0) {
            throw std::runtime_error("All tensors must be on CUDA");
        }

        const int B = static_cast<int>(bitset.shape(0));
        const int H = static_cast<int>(bitset.shape(1));
        const int W = static_cast<int>(bitset.shape(2));
        const int num_slots = static_cast<int>(bitset.shape(3));

        if (count_image.shape(0) != B || count_image.shape(1) != H || count_image.shape(2) != W) {
            throw std::runtime_error("count_image must have shape [B, H, W]");
        }

        strns2cards::cuda::popcount_bitset(
            B, H, W, num_slots,
            bitset.data(),
            count_image.data()
        );
    }, "Compute number of active bits per pixel in bitset image");

    m.def("dda_compute_span", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> edge_ids,         // [num_prims]
        int H, int W,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,         // [B, V, 4]
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> edges,         // [E, 2]
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> frag_prefix_sum,  // [num_prims]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> frag_slopes,     // [num_prims, 2]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_spans       // [num_prims, 4]
    ) -> int {
        if (edge_ids.device_id() < 0 || pos.device_id() < 0 || edges.device_id() < 0 ||
            frag_prefix_sum.device_id() < 0 || frag_slopes.device_id() < 0 || frag_spans.device_id() < 0) {
            throw std::runtime_error("All tensors must be CUDA tensors.");
        }

        const int num_prims = static_cast<int>(edge_ids.shape(0));
        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int E = static_cast<int>(edges.shape(0));

        if (frag_prefix_sum.shape(0) != num_prims ||
            frag_slopes.shape(0) != num_prims ||
            frag_spans.shape(0) != num_prims) {
            throw std::runtime_error("Invalid output tensor shapes for frag_prefix_sum, frag_slopes, or frag_spans.");
        }

        return strns2cards::cuda::dda_compute_span(
            num_prims,
            edge_ids.data(), H, W, V,
            pos.data(), E,
            edges.data(),
            frag_prefix_sum.data(),
            frag_slopes.data(),
            frag_spans.data()
        );
    }, "Compute fragment spans and slopes for each edge using DDA.");

    m.def("dda_compute_fragments", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> frag_prefix_sum, // [num_prims]
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> edge_ids,        // [num_prims]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> frag_slopes,    // [num_prims, 2]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_spans,     // [num_prims, 4]
        int H, int W,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,        // [B, V, 4]
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> edges,        // [E, 2]
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix,     // [num_frags, 3]
        nb::ndarray<float, nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs      // [num_frags, 4]
    ) {
        if (frag_prefix_sum.device_id() < 0 || edge_ids.device_id() < 0 ||
            frag_slopes.device_id() < 0 || frag_spans.device_id() < 0 ||
            pos.device_id() < 0 || edges.device_id() < 0 ||
            frag_pix.device_id() < 0 || frag_attrs.device_id() < 0) {
            throw std::runtime_error("All tensors must be CUDA tensors.");
        }

        const int num_frags = static_cast<int>(frag_pix.shape(0));
        const int num_prims = static_cast<int>(edge_ids.shape(0));
        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int E = static_cast<int>(edges.shape(0));

        if (frag_prefix_sum.shape(0) != num_prims ||
            frag_slopes.shape(0) != num_prims ||
            frag_spans.shape(0) != num_prims ||
            frag_attrs.shape(0) != num_frags) {
            throw std::runtime_error("Invalid shape for input or output arrays.");
        }

        strns2cards::cuda::dda_compute_fragments(
            num_prims,
            num_frags,
            frag_prefix_sum.data(),
            edge_ids.data(),
            frag_slopes.data(),
            frag_spans.data(),
            H, W, V,
            pos.data(),
            E, edges.data(),
            frag_pix.data(),
            frag_attrs.data()
        );
    }, "Generate pixel fragments along edges using DDA spans and slopes.");

    m.def("dda_interpolate_attributes", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>, nb::c_contig> rast_dda,  // [B, H, W, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1>,       nb::c_contig> attr,       // [V, C]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 2>,        nb::c_contig> edges,      // [E, 2]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> image     // [B, H, W, C]
    ) {
        // Device checks
        if (rast_dda.device_id() < 0 || attr.device_id() < 0 ||
            edges.device_id() < 0 || image.device_id() < 0) {
            throw std::runtime_error("All arrays must be CUDA tensors.");
        }

        // Extract dimensions
        const int B = static_cast<int>(rast_dda.shape(0));
        const int H = static_cast<int>(rast_dda.shape(1));
        const int W = static_cast<int>(rast_dda.shape(2));
        const int C = static_cast<int>(attr.shape(1));
        const int V = static_cast<int>(attr.shape(0));
        const int E = static_cast<int>(edges.shape(0));

        // Shape checks
        if (image.shape(0) != B || image.shape(1) != H || image.shape(2) != W || image.shape(3) != C)
            throw std::runtime_error("image must have shape [B, H, W, C]");

        // Flatten 2D access
        const float* rast_ptr = rast_dda.data();
        const float* attr_ptr = attr.data();
        const int*   edge_ptr = edges.data();
        float*       image_ptr = image.data();

        strns2cards::cuda::dda_interpolate_attributes(
            B, H, W, C,
            rast_ptr, attr_ptr, edge_ptr, image_ptr
        );
    }, "Interpolate per-vertex attributes for DDA fragments using CUDA");

    m.def("backward_dda_interpolate_attributes", [](
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, 4>,  nb::c_contig> rast_dda,   // [B, H, W, 4]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> d_image,    // [B, H, W, C]
        nb::ndarray<int32_t,  nb::pytorch, nb::shape<-1, 2>,          nb::c_contig> edges,      // [E, 2]
        nb::ndarray<float,    nb::pytorch, nb::shape<-1, -1>,         nb::c_contig> d_attr      // [V, C]
    ) {
        if (rast_dda.device_id() < 0 || d_image.device_id() < 0 ||
            edges.device_id() < 0 || d_attr.device_id() < 0) {
            throw std::runtime_error("All tensors must be on CUDA.");
        }

        const int B = static_cast<int>(rast_dda.shape(0));
        const int H = static_cast<int>(rast_dda.shape(1));
        const int W = static_cast<int>(rast_dda.shape(2));
        const int C = static_cast<int>(d_image.shape(3));

        if (d_image.shape(0) != B || d_image.shape(1) != H || d_image.shape(2) != W) {
            throw std::runtime_error("`d_image` must have shape [B, H, W, C] matching rast_dda");
        }
        if (d_attr.shape(1) != C) {
            throw std::runtime_error("`d_attr` must have shape [V, C] where C matches d_image.shape(3)");
        }

        strns2cards::cuda::backward_dda_interpolate_attributes(
            B, H, W,
            rast_dda.data(),
            C,
            d_image.data(),
            edges.data(),
            d_attr.data()
        );
    }, "Backward pass for interpolating per-pixel gradients to edge vertex attributes");

    m.def("mark_discontinuity_edges", [](
        int H, int W,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 3>, nb::c_contig> tri,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> edges,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> edge2tri,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> prim_ids,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, 2>, nb::c_contig> normals
    ) {
        if (pos.device_id() < 0 || tri.device_id() < 0 || edges.device_id() < 0 || edge2tri.device_id() < 0 ||
            prim_ids.device_id() < 0 || normals.device_id() < 0)
            throw std::runtime_error("mark_discontinuity_edges: all input tensors must be CUDA.");

        const int B = static_cast<int>(pos.shape(0));
        const int V = static_cast<int>(pos.shape(1));
        const int E = static_cast<int>(edges.shape(0));

        if (edge2tri.shape(0) != E)
            throw std::runtime_error("mark_discontinuity_edges: edge2tri must have the same number of edges.");
        if (prim_ids.shape(0) != B * E)
            throw std::runtime_error("mark_discontinuity_edges: prim_ids must have shape (B * E).");
        if (normals.shape(0) != B * E)
            throw std::runtime_error("mark_discontinuity_edges: normals must have shape (B * E, 2).");

        strns2cards::cuda::mark_discontinuity_edges(
            H, W, B,
            V, pos.data(),
            tri.data(),
            E, edges.data(),
            edge2tri.data(),
            prim_ids.data(),
            normals.data()
        );
    }, "Mark discontinuity edges based on triangle topology and vertex projection.");

    m.def("backward_antialiased_cluster_bitset", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 3>, nb::c_contig> frag_pix,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, 4>, nb::c_contig> frag_attrs_dda,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> bitset,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, -1, -1, -1>, nb::c_contig> target_bitset,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> pos,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1, 2>, nb::c_contig> edges,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, 2>, nb::c_contig> normals,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> edge2cluster,
        nb::ndarray<float,   nb::pytorch, nb::shape<-1, -1, 4>, nb::c_contig> d_pos,
        float kernel_radius,
        float rho
    ) {
        if (frag_pix.device_id() < 0 || frag_attrs_dda.device_id() < 0 ||
            bitset.device_id() < 0 || target_bitset.device_id() < 0 ||
            pos.device_id() < 0 || edges.device_id() < 0 ||
            normals.device_id() < 0 || edge2cluster.device_id() < 0 || d_pos.device_id() < 0)
            throw std::runtime_error("backward_antialiased_bitset: all input tensors must be CUDA.");

        const int num_frags = static_cast<int>(frag_pix.shape(0));
        const int B = static_cast<int>(bitset.shape(0));
        const int H = static_cast<int>(bitset.shape(1));
        const int W = static_cast<int>(bitset.shape(2));
        const int num_slots = static_cast<int>(bitset.shape(3));
        const int V = static_cast<int>(pos.shape(1));
        const int E = static_cast<int>(edges.shape(0));

        if (frag_attrs_dda.shape(0) != num_frags)
            throw std::runtime_error("backward_antialiased_bitset: frag_pix and frag_attrs_dda must match in number of fragments.");

        if (target_bitset.shape(0) != B ||
            target_bitset.shape(1) != H ||
            target_bitset.shape(2) != W ||
            target_bitset.shape(3) != num_slots)
            throw std::runtime_error("backward_antialiased_bitset: target_bitset must match bitset shape.");

        if (d_pos.shape(0) != B || d_pos.shape(1) != V)
            throw std::runtime_error("backward_antialiased_bitset: d_pos must match pos shape.");

        strns2cards::cuda::backward_antialiased_cluster_bitset(
            num_frags,
            frag_pix.data(),
            frag_attrs_dda.data(),
            H, W,
            num_slots,
            bitset.data(),
            target_bitset.data(),
            V, pos.data(),
            E, edges.data(),
            normals.data(),
            edge2cluster.data(),
            d_pos.data(),
            kernel_radius,
            rho
        );
    }, "Backward gradient computation for anti-aliased bitset optimization.");

    m.def("compact_valid_ints", [](
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> input,
        nb::ndarray<int32_t, nb::pytorch, nb::shape<-1>, nb::c_contig> output
    ) -> int {
        if (input.device_id() < 0 || output.device_id() < 0) {
            throw std::runtime_error("compact_valid_ints: both input and output must be CUDA tensors.");
        }
        if (input.shape(0) != output.shape(0)) {
            throw std::runtime_error("compact_valid_ints: input and output must have the same size.");
        }
        const int N = static_cast<int>(input.shape(0));
        return strns2cards::cuda::compact_valid_ints(N, input.data(), output.data());
    }, "Compact valid (≥0) integers from input array into output array.");

    m.def("load_strands", [](const char* filepath_cstr) -> auto {
        const std::string filepath(filepath_cstr);

        strns2cards::StrandsPoints strands;
        if (ends_with(filepath, ".bin")) {
            strands = strns2cards::load_bin(filepath);
        } else if (ends_with(filepath, ".data")) {
            strands = strns2cards::load_usc_data(filepath);
        } else if (ends_with(filepath, ".hair")) {
            strands = strns2cards::load_hair(filepath);
        } else {
            throw std::invalid_argument("load_strands: unknown file extension.");
        }

        size_t num_strands, num_points;
        int* num_samples_ptr = nullptr;
        float* points_ptr = nullptr;
        strns2cards::serialize_strands(strands, num_strands, num_samples_ptr, num_points, points_ptr);

        auto num_samples = nb::ndarray<nb::numpy, int32_t, nb::shape<-1, 1>, nb::c_contig>(
            num_samples_ptr, {num_strands, 1}, {}
        );
        auto points = nb::ndarray<nb::numpy, float, nb::shape<-1, 3>, nb::c_contig>(
            points_ptr, {num_points, 3}, {}
        );

        return nb::make_tuple(num_samples, points);
    }, "Load hair strands and return (num_samples, points) pair.",
    nb::rv_policy::take_ownership);

    m.def("resample_strands_by_arclength", [](
        nb::ndarray<int32_t, nb::numpy, nb::shape<-1>, nb::c_contig> num_samples,
        nb::ndarray<float, nb::numpy, nb::shape<-1, 3>, nb::c_contig> points,
        int target_num_samples
    ) -> auto {
        const size_t num_strands = num_samples.shape(0);
        const size_t num_points = points.shape(0);

        auto* out_ptr = new float[num_strands * target_num_samples * 3];
        strns2cards::resample_strands_by_arclength(
            static_cast<int>(num_strands),
            num_samples.data(),
            points.data(),
            target_num_samples,
            out_ptr
        );

        return nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 3>, nb::c_contig>(
            out_ptr, {num_strands, static_cast<size_t>(target_num_samples), 3}, {}
        );
    }, "Resample strands uniformly by arc length.",
    nb::rv_policy::take_ownership);

    m.def("smooth_strands", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3>, nb::c_contig> strands,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>,    nb::c_contig> arclength,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3>, nb::c_contig> smoothed_strands,
        float beta
    ) {
        if (strands.device_id() < 0 || arclength.device_id() < 0 || smoothed_strands.device_id() < 0) {
            throw std::runtime_error("smooth_strands: All input tensors must be CUDA.");
        }

        const int num_strands = static_cast<int>(strands.shape(0));
        const int num_points  = static_cast<int>(strands.shape(1));

        // Shape validation
        if (arclength.shape(0) != num_strands || arclength.shape(1) != num_points) {
            throw std::runtime_error("smooth_strands: arclength must match (num_strands, num_points).");
        }

        if (smoothed_strands.shape(0) != num_strands || smoothed_strands.shape(1) != num_points) {
            throw std::runtime_error("smooth_strands: smoothed_strands must match (num_strands, num_points, 3).");
        }

        strns2cards::cuda::smooth_strands(
            num_strands,
            num_points,
            strands.data(),
            arclength.data(),
            smoothed_strands.data(),
            beta
        );
    }, "Applies Gaussian smoothing to strands based on arclength distances.");

    m.def("skinning", [](
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3>,    nb::c_contig> strands,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1>,       nb::c_contig> arclength,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3>,    nb::c_contig> handle_strands,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3>,    nb::c_contig> canonical_handles,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3, 3>, nb::c_contig> R,
        nb::ndarray<float, nb::pytorch, nb::shape<-1, -1, 3>,    nb::c_contig> skinned_strands,
        float beta
    ) {
        if (strands.device_id() < 0 || arclength.device_id() < 0 || 
            handle_strands.device_id() < 0 || canonical_handles.device_id() < 0 || 
            R.device_id() < 0 || skinned_strands.device_id() < 0) {
            throw std::runtime_error("skinning: All input tensors must be CUDA.");
        }

        const int num_strands = static_cast<int>(strands.shape(0));
        const int num_points  = static_cast<int>(strands.shape(1));

        // Shape validation
        if (arclength.shape(0) != num_strands || arclength.shape(1) != num_points) {
            throw std::runtime_error("skinning: arclength must match (num_strands, num_points).");
        }

        if (handle_strands.shape(0) != num_strands || handle_strands.shape(1) != num_points) {
            throw std::runtime_error("skinning: handle_strands must match (num_strands, num_points, 3).");
        }

        if (canonical_handles.shape(0) != num_strands || canonical_handles.shape(1) != num_points) {
            throw std::runtime_error("skinning: canonical_handles must match (num_strands, num_points, 3).");
        }

        if (R.shape(0) != num_strands || R.shape(1) != num_points) {
            throw std::runtime_error("skinning: R must match (num_strands, num_points, 3, 3).");
        }

        if (skinned_strands.shape(0) != num_strands || skinned_strands.shape(1) != num_points) {
            throw std::runtime_error("skinning: skinned_strands must match (num_strands, num_points, 3).");
        }

        strns2cards::cuda::skinning(
            num_strands, 
            num_points,
            strands.data(), 
            arclength.data(),
            handle_strands.data(), 
            canonical_handles.data(), 
            R.data(),
            skinned_strands.data(),
            beta
        );
    }, "Applies Gaussian-weighted skinning using precomputed arclength and rotation matrices.");
}
