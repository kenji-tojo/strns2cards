#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

namespace strns2cards {

struct Vec3f {
    float x, y, z;

    Vec3f() : x(0.f), y(0.f), z(0.f) {}
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3f operator-(const Vec3f& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    [[nodiscard]] float norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};

using StrandPoints = std::vector<Vec3f>;
using StrandsPoints = std::vector<std::vector<Vec3f>>;
StrandsPoints load_bin(const std::string& filepath);
StrandsPoints load_usc_data(const std::string& filepath);
StrandsPoints load_hair(const std::string& filepath);

void serialize_strands(
    const StrandsPoints& strands,
    size_t& num_strands,
    int*& num_samples,
    size_t& num_points,
    float*& points
);

void resample_strands_by_arclength(
    int num_strands,
    const int* num_samples,
    const float* points,
    int target_num_samples,
    float* out
);

} // namespace strns2cards
