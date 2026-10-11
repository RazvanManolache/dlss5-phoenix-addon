#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace phoenix_addon::phoenix {
// Inference canvases and the image they contain are separate: padding allows
// a validated native launch plan to process other aspect ratios without stretch.
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
    unsigned width = 0, height = 0;
    if (native_plans) {
        // Prefer the requested canvas shape, then the largest canvas within
        // the pixel budget. These are real plans, never invented launch sizes.
        const std::array<std::array<unsigned, 2>, 4> plans{
            {{{640, 384}}, {{1280, 720}}, {{1920, 1080}}, {{2560, 1080}}}};
        double best = -1e30;
        for (const auto& plan : plans) {
            const double pixels = double(plan[0]) * plan[1];
            if (pixels > area * 1.001 || pixels >= double(output_width) * output_height)
                continue;
            const double score = std::log(pixels / area) -
                                 4 * std::abs(std::log((double(plan[0]) / plan[1]) / ratio));
            if (score > best) {
                best = score;
                width = plan[0];
                height = plan[1];
            }
        }
    } else {
        // Portable runners and the DLL accept aligned canvases; the output
        // retains the game's size and aspect regardless of canvas selection.
        width = std::max(16u, unsigned(std::sqrt(area * ratio)) / 16 * 16);
        height = std::max(16u, unsigned(std::sqrt(area / ratio)) / 16 * 16);
        if (width > 8192 || height > 8192 ||
            uint64_t(width) * height >= uint64_t(output_width) * output_height)
            width = height = 0;
    }
    if (!width) {
        auto result = fit_resolution(output_width, output_height, output_width, output_height);
        result.fallback = true;
        return result;
    }
    return fit_resolution(output_width, output_height, width, height);
}
} // namespace phoenix_addon::phoenix
