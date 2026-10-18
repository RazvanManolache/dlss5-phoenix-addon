#pragma once
#include "phoenix.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace phoenix_cpp {
inline void plane_require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Plane>
inline size_t plane_stride(const Plane& plane, unsigned width, unsigned height,
                           unsigned channels) {
    plane_require(plane.data && !plane.reserved && width && height && width <= 8192 &&
                      height <= 8192 && plane.width == width && plane.height == height &&
                      plane.channels == channels,
                  "Invalid plane dimensions, channels or reserved fields");
    const uint64_t row = uint64_t(width) * channels * sizeof(float);
    const uint64_t stride = plane.row_stride_bytes ? plane.row_stride_bytes : row;
    plane_require(stride >= row && stride <= (UINT64_MAX - row) / height,
                  "Invalid plane row stride");
    const uint64_t required = stride * (height - 1) + row;
    plane_require(required <= plane.capacity_bytes && required <= SIZE_MAX &&
                      required <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(plane.data),
                  "Insufficient plane capacity or address overflow");
    return size_t(stride);
}
inline std::vector<float> copy_plane(const phoenix_plane_f32& plane, unsigned width,
                                     unsigned height, unsigned channels, bool required) {
    if (!plane.data && !required) {
        plane_require(!plane.capacity_bytes && !plane.row_stride_bytes && !plane.width &&
                          !plane.height && !plane.channels && !plane.reserved,
                      "Absent optional plane must be zero-initialized");
        return {};
    }
    const size_t stride = plane_stride(plane, width, height, channels);
    std::vector<float> result(size_t(width) * height * channels);
    const auto* bytes = reinterpret_cast<const unsigned char*>(plane.data);
    for (unsigned y = 0; y < height; ++y)
        std::memcpy(result.data() + size_t(y) * width * channels, bytes + size_t(y) * stride,
                    size_t(width) * channels * sizeof(float));
    return result;
}
inline void validate_control_mask(const float* values, size_t pixels) {
    if (!values) return;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        for (unsigned channel = 0; channel < 4; ++channel)
            plane_require(std::isfinite(values[pixel * 4 + channel]) &&
                              values[pixel * 4 + channel] >= 0 &&
                              values[pixel * 4 + channel] <= 1,
                          "Control mask requires finite linear values in 0..1");
        plane_require(values[pixel * 4 + 3] == 1.f,
                      "Experimental RGB mask requires reserved alpha = 1");
    }
}
}
