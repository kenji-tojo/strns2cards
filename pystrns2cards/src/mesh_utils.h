#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

namespace strns2cards {

void build_edges_from_triangles(
    int T,                          // Number of triangles
    const int* tri,                 // (T * 3,) triangle indices
    std::vector<int>& edges_out,    // (E * 2,) flattened output edges
    std::vector<int>& edge2tri_out  // (E * 2,) flattened output triangle mapping
);

} // namespace strns2cards
