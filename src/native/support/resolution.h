#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace phoenix_addon::phoenix {
// Inference canvases and the image they contain are separate: padding allows
// the chosen inference canvas to process other aspect ratios without stretch.
struct Resolution {
    unsigned output_width = 0, output_height = 0;
    unsigned width = 0, height = 0;
    unsigned image_width = 0, image_height = 0, left = 0, top = 0;
    bool fallback = false;
    bool scaled() const {
        return width != output_width || height != output_height;
    }
    float motion_x() const {
        return float(image_width) / output_width;
    }
    float motion_y() const {
        return float(image_height) / output_height;
    }
};
inline Resolution fit_resolution(unsigned output_width, unsigned output_height, unsigned width,
                                 unsigned height) {
    Resolution result{output_width, output_height, width, height};
    const double scale = std::min(double(width) / output_width, double(height) / output_height);
    result.image_width = std::max(1u, unsigned(std::round(output_width * scale)));
    result.image_height = std::max(1u, unsigned(std::round(output_height * scale)));
    result.left = (width - result.image_width) / 2;
    result.top = (height - result.image_height) / 2;
    return result;
}
inline Resolution choose_resolution(unsigned output_width, unsigned output_height, int mode,
                                    int quality, int aspect, int ratio_width, int ratio_height,
                                    bool native_plans) {
    if (!output_width || !output_height)
        return {};
    if (!mode)
        return fit_resolution(output_width, output_height, output_width, output_height);
    const double scale[] = {2.0 / 3.0, 0.5, 1.0 / 3.0};
    const double ratios[] = {double(output_width) / output_height,
                             16.0 / 9.0,
                             16.0 / 10.0,
                             21.0 / 9.0,
                             4.0 / 3.0,
                             double(std::max(1, ratio_width)) / std::max(1, ratio_height)};
    const double ratio = ratios[std::clamp(aspect, 0, 5)];
    const double area = double(output_width) * output_height * scale[std::clamp(quality, 0, 2)] *
                        scale[std::clamp(quality, 0, 2)];
    // Native plans pad feature tensors internally, so the active image can
    // use exact pixel dimensions. Portable/vendor routes retain their current
    // 16-pixel canvas alignment. Neither route chooses from fixed presets.
    const unsigned alignment = native_plans ? 1u : 16u;
    const unsigned maximum = native_plans ? 16384u : 8192u;
    unsigned width = std::max(alignment, unsigned(std::sqrt(area * ratio)) / alignment * alignment);
    unsigned height = std::max(alignment, unsigned(std::sqrt(area / ratio)) / alignment * alignment);
    if (width > maximum || height > maximum ||
        uint64_t(width) * height >= uint64_t(output_width) * output_height)
        width = height = 0;
    if (!width) {
        auto result = fit_resolution(output_width, output_height, output_width, output_height);
        result.fallback = true;
        return result;
    }
    return fit_resolution(output_width, output_height, width, height);
}
} // namespace phoenix_addon::phoenix
