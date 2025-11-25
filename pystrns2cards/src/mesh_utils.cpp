#include "mesh_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <unordered_map>

namespace strns2cards {

// Undirected edge key (sorted pair)
struct EdgeKey {
    int v0, v1;

    EdgeKey(int a, int b) {
        if (a < b) { v0 = a; v1 = b; }
        else       { v0 = b; v1 = a; }
    }

    bool operator==(const EdgeKey& other) const {
        return v0 == other.v0 && v1 == other.v1;
    }
};

// Hash function for EdgeKey
struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& e) const {
        return static_cast<std::size_t>(e.v0) * 73856093u ^ static_cast<std::size_t>(e.v1) * 19349663u;
    }
};

// Main function to extract edges and edge-to-triangle mapping
void build_edges_from_triangles(
    int T,                          // Number of triangles
    const int* tri,                 // (T * 3,) triangle indices
    std::vector<int>& edges_out,   // (E * 2,) flattened output edges
    std::vector<int>& edge2tri_out // (E * 2,) flattened output triangle mapping
) {
    std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edge_map;

    // Collect all triangle edges
    for (int t = 0; t < T; ++t) {
        const int i0 = tri[t * 3 + 0];
        const int i1 = tri[t * 3 + 1];
        const int i2 = tri[t * 3 + 2];

        edge_map[EdgeKey(i0, i1)].push_back(t);
        edge_map[EdgeKey(i1, i2)].push_back(t);
        edge_map[EdgeKey(i2, i0)].push_back(t);
    }

    edges_out.clear();
    edge2tri_out.clear();

    for (const auto& entry : edge_map) {
        const EdgeKey& edge = entry.first;
        const std::vector<int>& tri_ids = entry.second;

        // Add edge (flattened)
        edges_out.push_back(edge.v0);
        edges_out.push_back(edge.v1);

        // Add edge-to-tri mapping (up to 2 triangles)
        if (tri_ids.size() == 1) {
            edge2tri_out.push_back(tri_ids[0]);
            edge2tri_out.push_back(-1);
        } else if (tri_ids.size() >= 2) {
            edge2tri_out.push_back(tri_ids[0]);
            edge2tri_out.push_back(tri_ids[1]);
        } else {
            // Should never happen unless input is invalid
            edge2tri_out.push_back(-1);
            edge2tri_out.push_back(-1);
        }
    }
}

} // strns2cards
