#pragma once
#include "half_rgba.h"
#include "backend.h"
#include "host_math.h"
#include <algorithm>

namespace phoenix_portable_pixels {
inline void color(const uint16_t* source, float* rgb, uint16_t* rgba, size_t pixels) {
    std::copy_n(source, pixels * 4, rgba);
    phoenix_native_cuda::half_rgba_to_rgb(source, rgb, pixels);
    for (size_t i = 0; i < pixels * 3; ++i) rgb[i] = std::clamp(rgb[i], 0.f, 1.f);
}
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
__declspec(noinline) inline void motion_f16c(const uint16_t* source, float* output,
                                            size_t count, float sx, float sy) {
    const auto scale = _mm256_setr_ps(sx, sy, sx, sy, sx, sy, sx, sy);
    size_t i = 0;
    for (; i + 8 <= count; i += 8)
        _mm256_storeu_ps(output + i, _mm256_mul_ps(scale, _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i)))));
    for (; i < count; ++i) output[i] = phoenix_native_cuda::half_value(source[i]) * (i % 2 ? sy : sx);
}
// Conversion occurs only after the caller's finite-value check. Keep alpha's
// original bits, and explicitly round to nearest even independently of MXCSR.
__declspec(noinline) inline void output_f16c(const float* rgb, uint16_t* rgba, size_t pixels) {
    size_t p = 0;
    for (; p + 1 < pixels; p += 2) {
        const auto value = _mm256_setr_ps(rgb[p*3], rgb[p*3+1], rgb[p*3+2], 0.f,
            rgb[p*3+3], rgb[p*3+4], rgb[p*3+5], 0.f);
        alignas(16) uint16_t packed[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(packed), _mm256_cvtps_ph(value, 0));
        std::copy_n(packed, 3, rgba+p*4);
        std::copy_n(packed+4, 3, rgba+p*4+4);
    }
    phx::Kernel<HostMath> math{};
    for (; p < pixels; ++p) for (unsigned c=0;c<3;++c) rgba[p*4+c]=uint16_t(math.half_bits(rgb[p*3+c]));
}
#endif
inline void motion(const uint16_t* source, float* output, size_t count, float sx, float sy) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    if (phoenix_native_cuda::host_f16c_available()) return motion_f16c(source, output, count, sx, sy);
#endif
    for (size_t i=0;i<count;++i) output[i]=phoenix_native_cuda::half_value(source[i])*(i%2?sy:sx);
}
inline void output(const float* rgb, uint16_t* rgba, size_t pixels) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    if (phoenix_native_cuda::host_f16c_available()) return output_f16c(rgb, rgba, pixels);
#endif
    phx::Kernel<HostMath> math{};
    for (size_t p=0;p<pixels;++p) for (unsigned c=0;c<3;++c) rgba[p*4+c]=uint16_t(math.half_bits(rgb[p*3+c]));
}
}
