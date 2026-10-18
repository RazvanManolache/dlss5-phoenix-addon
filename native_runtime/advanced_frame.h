#pragma once
#include "appearance.h"
#include "frame_planes.h"

namespace phoenix_cpp {
struct AdvancedFrame : DisplayInputs {
    const float* distortion = nullptr;
};
inline void validate_unit_plane(const float* values, size_t count, const char* message) {
    if (!values) return;
    for (size_t i = 0; i < count; ++i)
        plane_require(std::isfinite(values[i]) && values[i] >= 0 && values[i] <= 1, message);
}
inline float plane_sample(const float* data, unsigned width, unsigned height, unsigned channels,
                          float x, float y, unsigned channel) {
    x = std::clamp(x, 0.f, float(width - 1));
    y = std::clamp(y, 0.f, float(height - 1));
    const unsigned left = unsigned(x), top = unsigned(y),
        right = std::min(left + 1, width - 1), bottom = std::min(top + 1, height - 1);
    const float fx = x - left, fy = y - top;
    const auto read = [&](unsigned sx, unsigned sy) { return data[(size_t(sy) * width + sx) * channels + channel]; };
    return (read(left, top) * (1-fx) + read(right, top) * fx) * (1-fy) +
        (read(left, bottom) * (1-fx) + read(right, bottom) * fx) * fy;
}
inline std::vector<float> warp_plane(const float* source, const float* field, unsigned width,
                                     unsigned height, unsigned channels, unsigned direction) {
    std::vector<float> result(size_t(width) * height * channels);
    for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
        const size_t pixel = size_t(y) * width + x;
        const float sx = x + field[pixel * 4 + direction] * width,
                    sy = y + field[pixel * 4 + direction + 1] * height;
        for (unsigned c = 0; c < channels; ++c)
            result[pixel * channels + c] = plane_sample(source, width, height, channels, sx, sy, c);
    }
    return result;
}
struct CanonicalFrame {
    std::vector<float> color, motion, depth, validity, mask;
    void prepare(const float* original, const float* vectors, const float* z, const float* valid,
                 const float* controls, const float* field, const float* previous_field,
                 unsigned width, unsigned height) {
        color = warp_plane(original, field, width, height, 3, 2);
        if (z) depth = warp_plane(z, field, width, height, 1, 2);
        if (controls) mask = warp_plane(controls, field, width, height, 4, 2);
        if (!vectors) return;
        motion.resize(size_t(width) * height * 2);
        validity.resize(size_t(width) * height);
        for (unsigned y = 0; y < height; ++y) for (unsigned x = 0; x < width; ++x) {
            const size_t pixel = size_t(y) * width + x;
            const float sx = x + field[pixel * 4 + 2] * width,
                        sy = y + field[pixel * 4 + 3] * height;
            const float px = sx + plane_sample(vectors, width, height, 2, sx, sy, 0),
                        py = sy + plane_sample(vectors, width, height, 2, sx, sy, 1);
            const float* prior_field = previous_field ? previous_field : field;
            const float cx = px + plane_sample(prior_field, width, height, 4, px, py, 0) * width,
                        cy = py + plane_sample(prior_field, width, height, 4, px, py, 1) * height;
            motion[pixel * 2] = cx - x;
            motion[pixel * 2 + 1] = cy - y;
            const bool inside = sx >= 0 && sy >= 0 && sx <= width-1 && sy <= height-1 &&
                px >= 0 && py >= 0 && px <= width-1 && py <= height-1;
            const unsigned nearest_x = unsigned(std::clamp(std::floor(sx+.5f), 0.f, float(width-1))),
                           nearest_y = unsigned(std::clamp(std::floor(sy+.5f), 0.f, float(height-1)));
            validity[pixel] = inside && (!valid || valid[size_t(nearest_y)*width+nearest_x] == 1.f);
        }
    }
};
}
