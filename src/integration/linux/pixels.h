// Phoenix's SDR proxy domain is nonlinear RGB in [0,1]. An sRGB swapchain copy
// contains encoded bytes too: do not apply the sRGB transfer a second time.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace phoenix_linux {
inline void unpack(const uint8_t* bytes, float* rgb, size_t pixels, bool bgra) {
    for (size_t i = 0; i < pixels; ++i) {
        rgb[i*3] = bytes[i*4 + (bgra ? 2 : 0)] / 255.f;
        rgb[i*3+1] = bytes[i*4+1] / 255.f;
        rgb[i*3+2] = bytes[i*4 + (bgra ? 0 : 2)] / 255.f;
    }
}
// Preserve the game's alpha, including compositor transparency.
inline void pack(const float* rgb, uint8_t* bytes, size_t pixels, bool bgra) {
    for (size_t i = 0; i < pixels; ++i)
        for (size_t c = 0; c < 3; ++c) {
            float v = rgb[i*3+c];
            if (!std::isfinite(v)) v = 0;
            // Round nonnegative SDR values exactly as lround, without a
            // library call per channel. Test the fractional part separately:
            // adding 0.5f first can misround values just below a half boundary.
            const float scaled = std::clamp(v, 0.f, 1.f) * 255.f;
            const unsigned whole = static_cast<unsigned>(scaled);
            bytes[i*4 + (bgra && c != 1 ? 2-c : c)] =
                uint8_t(whole + unsigned(scaled - float(whole) >= 0.5f));
        }
}
}
