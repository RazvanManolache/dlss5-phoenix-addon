#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace phoenix_cpp {
using RGB = std::array<float, 3>;
inline float saturate(float x) { return std::clamp(x, 0.f, 1.f); }
// Independently authored HSL saturation update; hue/lightness remain unchanged.
inline RGB scale_saturation(RGB rgb, float scale) {
    const float low = std::min({rgb[0], rgb[1], rgb[2]}),
                high = std::max({rgb[0], rgb[1], rgb[2]}),
                light = (high + low) * .5f, chroma = high - low;
    if (chroma <= 0) return rgb;
    const float denominator = 1.f - std::fabs(2.f * light - 1.f);
    const float old_saturation = denominator > 0 ? chroma / denominator : 0;
    const float factor = old_saturation > 0 ? saturate(old_saturation * scale) / old_saturation : 0;
    for (auto& channel : rgb) channel = saturate(light + (channel - light) * factor);
    return rgb;
}
// Edition 310.8's preset data was independently checked in the selected DLL.
// The host arithmetic/activation remains a recovered contract, not certification.
inline RGB style_grade(RGB rgb, unsigned style, float tone) {
    const float amount = saturate(tone);
    if (style == 1) {
        const float exposure = std::exp2(-.1f * amount), contrast = -.25f * amount;
        for (auto& channel : rgb) {
            channel = saturate(channel * exposure);
            channel = saturate(channel + contrast *
                (channel * channel * (3.f - 2.f * channel) - channel));
        }
        return scale_saturation(rgb, 1.f - .1f * amount);
    }
    return style == 2 ? scale_saturation(rgb, 1.f - .15f * amount) : rgb;
}
struct DisplayInputs {
    const float* control_mask = nullptr; // RGBA, full active extent
    const float* ui = nullptr;           // premultiplied RGBA overlay
    const float* ui_alpha = nullptr;     // coverage overrides overlay alpha
    const float* backbuffer = nullptr;   // RGB base already containing UI
};
inline RGB compose_display(RGB original, RGB neural, size_t pixel, float intensity,
                           unsigned style, float tone, const DisplayInputs& inputs) {
    const float alpha = inputs.ui_alpha ? inputs.ui_alpha[pixel]
        : inputs.ui ? inputs.ui[pixel * 4 + 3] : 0.f;
    RGB base = original;
    if (inputs.backbuffer)
        for (unsigned c = 0; c < 3; ++c) base[c] = inputs.backbuffer[pixel * 3 + c];
    else if (inputs.ui)
        for (unsigned c = 0; c < 3; ++c)
            base[c] = inputs.ui[pixel * 4 + c] + original[c] * (1.f - alpha);
    const bool postprocess = intensity < 1.f || inputs.control_mask || inputs.ui ||
        inputs.ui_alpha || inputs.backbuffer || (style && tone > 0);
    if (!postprocess) return neural; // edition's >1-only path skips postprocess
    const auto graded = style_grade(neural, style, tone);
    const float weight = intensity * (inputs.control_mask ? inputs.control_mask[pixel * 4] : 1.f)
        * (1.f - alpha);
    for (unsigned c = 0; c < 3; ++c) base[c] = saturate(base[c] + weight * (graded[c] - original[c]));
    return base;
}
}
