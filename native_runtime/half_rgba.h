#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PHOENIX_HAS_F16C_DISPATCH 1
#endif
namespace phoenix_native_cuda {
inline float half_value(uint16_t half) {
    const uint32_t sign = uint32_t(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 31u, fraction = half & 1023u;
    if (!exponent) {
        if (!fraction)
            return std::bit_cast<float>(sign);
        int shift = 0;
        while (!(fraction & 1024u)) {
            fraction <<= 1;
            ++shift;
        }
        return std::bit_cast<float>(sign | (uint32_t(113 - shift) << 23) |
                                    ((fraction & 1023u) << 13));
    }
    return std::bit_cast<float>(sign | ((exponent == 31 ? 255u : exponent + 112) << 23) |
                                (fraction << 13));
}
inline void half_rgba_scalar(const uint16_t* source, float* target, size_t pixels) {
    for (size_t pixel = 0; pixel < pixels; ++pixel)
        for (unsigned channel = 0; channel < 3; ++channel)
            target[pixel * 3 + channel] = half_value(source[pixel * 4 + channel]);
}
#ifdef PHOENIX_HAS_F16C_DISPATCH
__attribute__((target("avx,f16c"))) inline void half_rgba_f16c(const uint16_t* source,
                                                               float* target, size_t pixels) {
    size_t pixel = 0;
    for (; pixel + 1 < pixels; pixel += 2) {
        alignas(32) float rgba[8];
        const auto packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + pixel * 4));
        _mm256_store_ps(rgba, _mm256_cvtph_ps(packed));
        target[pixel * 3] = rgba[0];
        target[pixel * 3 + 1] = rgba[1];
        target[pixel * 3 + 2] = rgba[2];
        target[pixel * 3 + 3] = rgba[4];
        target[pixel * 3 + 4] = rgba[5];
        target[pixel * 3 + 5] = rgba[6];
    }
    half_rgba_scalar(source + pixel * 4, target + pixel * 3, pixels - pixel);
}
#endif
inline void half_rgba_to_rgb(const uint16_t* source, float* target, size_t pixels) {
#ifdef PHOENIX_HAS_F16C_DISPATCH
    // Runtime checks preserve the baseline binary's ISA requirements.
    static const bool supported = __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
    if (supported) {
        half_rgba_f16c(source, target, pixels);
        return;
    }
#endif
    half_rgba_scalar(source, target, pixels);
}
} // namespace phoenix_native_cuda
#undef PHOENIX_HAS_F16C_DISPATCH
