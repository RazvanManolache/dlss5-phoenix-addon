// Graphics rows are normalized to top-down RGBA8 for the helper.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "../../../third_party/reshade/reshade_api.hpp"
namespace phoenix_compat {
inline bool uses_buffer_readback(reshade::api::device_api api) {
    using reshade::api::device_api;
    return api == device_api::d3d12 || api == device_api::vulkan || api == device_api::opengl;
}
inline bool uses_helper(reshade::api::device_api api, unsigned process_bits) {
    return process_bits == 32 || api != reshade::api::device_api::d3d12;
}
inline unsigned transfer_row_pitch(reshade::api::device_api api, unsigned width) {
    if (!width || width > 8192)
        throw std::runtime_error("Unsupported transfer width");
    const unsigned bytes = width * 4;
    return api == reshade::api::device_api::d3d12 ? (bytes + 255) & ~255u : bytes;
}
inline void copy_pixel_rows(uint8_t* destination, unsigned destination_pitch, const uint8_t* source,
                            unsigned source_pitch, unsigned width, unsigned height, bool flip_rows,
                            bool bgra) {
    if (!destination || !source || !height || destination_pitch < width * 4 ||
        source_pitch < width * 4)
        throw std::runtime_error("Invalid graphics row layout");
    for (unsigned row = 0; row < height; ++row) {
        auto* output = destination + size_t(row) * destination_pitch;
        const auto* input = source + size_t(flip_rows ? height - 1 - row : row) * source_pitch;
        std::memcpy(output, input, size_t(width) * 4);
        if (bgra)
            for (unsigned column = 0; column < width; ++column)
                std::swap(output[column * 4], output[column * 4 + 2]);
    }
}
} // namespace phoenix_compat
