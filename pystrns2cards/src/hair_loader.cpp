#include "hair_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <unordered_map>

namespace strns2cards {

// --- Loader for general .bin format ---
StrandsPoints load_bin(const std::string& filepath) {
    StrandsPoints strands_points;
#ifdef _WIN32
    FILE* f;
    fopen_s(&f, filepath.c_str(), "rb");
#else
    FILE* f = fopen(filepath.c_str(), "rb");
#endif
    if (!f) {
        fprintf(stderr, "Couldn't open %s\n", filepath.c_str());
        return strands_points;
    }

    int num_strands = 0;
    fread(&num_strands, 4, 1, f);
    for (int i = 0; i < num_strands; ++i) {
        int num_points = 0;
        fread(&num_points, 4, 1, f);
        std::vector<Vec3f> strand(num_points);

        float dummy;
        for (int j = 0; j < num_points; ++j) {
            fread(&strand[j].x, 4, 1, f);
            fread(&strand[j].y, 4, 1, f);
            fread(&strand[j].z, 4, 1, f);
            fread(&dummy, 4, 1, f); // nx
            fread(&dummy, 4, 1, f); // ny
            fread(&dummy, 4, 1, f); // nz
            fread(&dummy, 4, 1, f); // label
        }

        bool valid = true;
        for (int j = 0; j < num_points - 1; ++j) {
            if ((strand[j] - strand[j + 1]).norm() < 1e-4f) {
                valid = false;
                break;
            }
        }

        if (valid) {
            strands_points.push_back(std::move(strand));
        }
    }
    fclose(f);
    return strands_points;
}

// --- Loader for USC dataset format ---
StrandsPoints load_usc_data(const std::string& filepath) {
    StrandsPoints strands_points;
#ifdef _WIN32
    FILE* f;
    fopen_s(&f, filepath.c_str(), "rb");
#else
    FILE* f = fopen(filepath.c_str(), "rb");
#endif
    if (!f) {
        fprintf(stderr, "Couldn't open %s\n", filepath.c_str());
        return strands_points;
    }

    int num_strands = 0;
    fread(&num_strands, 4, 1, f);
    for (int i = 0; i < num_strands; ++i) {
        int num_points;
        fread(&num_points, 4, 1, f);
        if (num_points == 1) {
            float dummy[3];
            fread(dummy, 4, 3, f);
        } else {
            std::vector<Vec3f> strand(num_points);
            for (int j = 0; j < num_points; ++j) {
                fread(&strand[j].x, 4, 1, f);
                fread(&strand[j].y, 4, 1, f);
                fread(&strand[j].z, 4, 1, f);
            }
            strands_points.push_back(std::move(strand));
        }
    }
    fclose(f);
    return strands_points;
}

// --- Loader for HAIR/ZJU datasets ---
StrandsPoints load_hair(const std::string& filepath) {
    StrandsPoints strands_points;

#ifdef _WIN32
    FILE* f;
    fopen_s(&f, filepath.c_str(), "rb");
#else
    FILE* f = fopen(filepath.c_str(), "rb");
#endif
    if (!f) {
        fprintf(stderr, "Couldn't open %s\n", filepath.c_str());
        return strands_points;
    }

    // --- Read header to detect file type ---
    char header[4];
    if (fread(header, 1, 4, f) != 4) {
        fprintf(stderr, "Failed to read header.\n");
        fclose(f);
        return strands_points;
    }

    bool is_hair_format = (header[0] == 'H' && header[1] == 'A' && header[2] == 'I' && header[3] == 'R');

    if (is_hair_format) {
        // --- HAIR format ---
        unsigned int num_strands = 0;
        unsigned int points_count = 0;
        unsigned int bit_array = 0;
        unsigned int default_num_seg = 0;

        fread(&num_strands, sizeof(unsigned int), 1, f);
        fread(&points_count, sizeof(unsigned int), 1, f);
        fread(&bit_array, sizeof(unsigned int), 1, f);

        constexpr unsigned int cy_hair_file_segments_bit = 1;
        constexpr unsigned int cy_hair_file_points_bit = 2;

        if (!(bit_array & cy_hair_file_points_bit)) {
            fprintf(stderr, "Error: HAIR file missing points section.\n");
            fclose(f);
            return strands_points;
        }
        if (!(bit_array & cy_hair_file_segments_bit)) {
            fread(&default_num_seg, sizeof(unsigned int), 1, f);
        }

        // Skip rest of header (fixed size: 128 unsigned ints total)
        fseek(f, sizeof(unsigned int) * (128 - (default_num_seg ? 3 : 4)), SEEK_CUR);

        // --- Allocate and read data ---
        auto* segments = new unsigned short[num_strands];
        auto* points = new float[points_count * 3];

        if (default_num_seg == 0) {
            fread(segments, sizeof(unsigned short), num_strands, f);
        } else {
            for (unsigned int i = 0; i < num_strands; ++i) {
                segments[i] = static_cast<unsigned short>(default_num_seg);
            }
        }
        fread(points, sizeof(float), points_count * 3, f);

        // --- Fill output ---
        int point_idx = 0;
        for (unsigned int i = 0; i < num_strands; ++i) {
            int num_points = segments[i] + 1;
            StrandPoints strand(num_points);
            for (int j = 0; j < num_points; ++j) {
                strand[j] = Vec3f(
                    points[(point_idx + j) * 3 + 0],
                    points[(point_idx + j) * 3 + 1],
                    points[(point_idx + j) * 3 + 2]
                );
            }
            strands_points.push_back(std::move(strand));
            point_idx += num_points;
        }

        delete[] segments;
        delete[] points;
    } else {
        // --- ZJU format ---
        // Reset file pointer to start
        fseek(f, 0, SEEK_SET);

        unsigned int num_strands = 0;
        unsigned int points_count = 0;
        fread(&num_strands, sizeof(unsigned int), 1, f);
        fread(&points_count, sizeof(unsigned int), 1, f);

        auto* segments = new unsigned short[num_strands];
        auto* points = new float[points_count * 3];

        fread(segments, sizeof(unsigned short), num_strands, f);
        fread(points, sizeof(float), points_count * 3, f);

        int point_idx = 0;
        for (unsigned int i = 0; i < num_strands; ++i) {
            int num_points = segments[i];
            StrandPoints strand(num_points);
            for (int j = 0; j < num_points; ++j) {
                strand[j] = Vec3f(
                    points[(point_idx + j) * 3 + 0],
                    points[(point_idx + j) * 3 + 1],
                    points[(point_idx + j) * 3 + 2]
                );
            }
            strands_points.push_back(std::move(strand));
            point_idx += num_points;
        }

        delete[] segments;
        delete[] points;
    }

    fclose(f);
    return strands_points;
}

// --- Flatten strands into contiguous arrays for batch processing ---
void serialize_strands(
    const StrandsPoints& strands,
    size_t& num_strands,
    int*& num_samples,
    size_t& num_points,
    float*& points
) {
    num_strands = 0;
    std::vector<int> samples_tmp;
    samples_tmp.reserve(strands.size());

    for (const auto& strand : strands) {
        if (strand.size() >= 2) {
            ++num_strands;
            samples_tmp.push_back(static_cast<int>(strand.size()));
        }
    }

    num_samples = new int[samples_tmp.size()];
    std::copy(samples_tmp.begin(), samples_tmp.end(), num_samples);

    num_points = std::accumulate(samples_tmp.begin(), samples_tmp.end(), size_t(0));
    points = new float[num_points * 3];

    size_t idx = 0;
    for (const auto& strand : strands) {
        if (strand.size() >= 2) {
            for (const auto& p : strand) {
                points[idx++] = p.x;
                points[idx++] = p.y;
                points[idx++] = p.z;
            }
        }
    }

    if (idx != num_points * 3) {
        std::fprintf(stderr, "Error: serialize_strands mismatch in point count\n");
    }
}

// --- Resample strands uniformly based on arc length ---
void resample_strands_by_arclength(
    int num_strands,
    const int* num_samples,
    const float* points,
    int target_num_samples,
    float* out
) {
    size_t point_idx = 0;
    for (int i = 0; i < num_strands; ++i) {
        int N = num_samples[i];
        const float* p = points + point_idx * 3;

        std::vector<float> length(N, 0.f);
        for (int j = 1; j < N; ++j) {
            float dx = p[j * 3 + 0] - p[(j - 1) * 3 + 0];
            float dy = p[j * 3 + 1] - p[(j - 1) * 3 + 1];
            float dz = p[j * 3 + 2] - p[(j - 1) * 3 + 2];
            length[j] = length[j - 1] + std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        for (float& l : length) {
            l /= length.back();
        }

        float* out_ptr = out + i * target_num_samples * 3;
        for (int j = 0; j < target_num_samples - 1; ++j) {
            float t = static_cast<float>(j) / static_cast<float>(target_num_samples - 1);
            auto it = std::upper_bound(length.begin(), length.end(), t);
            int idx = static_cast<int>(std::distance(length.begin(), it));
            idx = std::min(std::max(idx, 1), N - 1);

            float t0 = (length[idx] - t) / (length[idx] - length[idx - 1]);
            float t1 = (t - length[idx - 1]) / (length[idx] - length[idx - 1]);

            out_ptr[j * 3 + 0] = t0 * p[(idx - 1) * 3 + 0] + t1 * p[idx * 3 + 0];
            out_ptr[j * 3 + 1] = t0 * p[(idx - 1) * 3 + 1] + t1 * p[idx * 3 + 1];
            out_ptr[j * 3 + 2] = t0 * p[(idx - 1) * 3 + 2] + t1 * p[idx * 3 + 2];
        }

        // Last point (exact copy)
        out_ptr[(target_num_samples - 1) * 3 + 0] = p[(N - 1) * 3 + 0];
        out_ptr[(target_num_samples - 1) * 3 + 1] = p[(N - 1) * 3 + 1];
        out_ptr[(target_num_samples - 1) * 3 + 2] = p[(N - 1) * 3 + 2];

        point_idx += N;
    }
}

} // namespace strns2cards
