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
        if (!bgra) {
            std::memcpy(output, input, size_t(width) * 4);
        } else {
            // Upload memory can be write-combined device memory. Reading it
            // back to swap channels turns every pixel into a slow PCIe read.
            // Swizzle in a CPU register and write the destination exactly once.
            for (unsigned column = 0; column < width; ++column) {
                uint32_t pixel;
                std::memcpy(&pixel, input + column * 4, sizeof(pixel));
                pixel = (pixel & 0xff00ff00u) | ((pixel & 0xffu) << 16) | ((pixel >> 16) & 0xffu);
                std::memcpy(output + column * 4, &pixel, sizeof(pixel));
            }
        }
    }
}
// The helper currently carries RGBA8 SDR. Preserve the swapchain's packed
// format at both GPU boundaries; never interpret RGB10A2 words as RGBA bytes.
inline void copy_rgb10a2_rows(uint8_t* destination, unsigned destination_pitch,
                            const uint8_t* source, unsigned source_pitch,
                            unsigned width, unsigned height, bool decode) {
    if (!destination || !source || !width || width > 8192 || !height ||
        destination_pitch < width * 4 || source_pitch < width * 4)
        throw std::runtime_error("Invalid RGB10A2 row layout");
    for (unsigned row = 0; row < height; ++row)
        for (unsigned column = 0; column < width; ++column) {
            auto* output = destination + size_t(row) * destination_pitch + column * 4;
            const auto* input = source + size_t(row) * source_pitch + column * 4;
            if (decode) {
                uint32_t packed = 0;
                std::memcpy(&packed, input, sizeof(packed));
                for (unsigned channel = 0; channel < 3; ++channel)
                    output[channel] = uint8_t((((packed >> (channel * 10)) & 1023u) * 255u + 511u) / 1023u);
                output[3] = uint8_t((packed >> 30) * 85u);
            } else {
                uint32_t packed = 0;
                for (unsigned channel = 0; channel < 3; ++channel)
                    packed |= ((uint32_t(input[channel]) * 1023u + 127u) / 255u) << (channel * 10);
                packed |= ((uint32_t(input[3]) * 3u + 127u) / 255u) << 30;
                std::memcpy(output, &packed, sizeof(packed));
            }
        }
}
} // namespace phoenix_compat
