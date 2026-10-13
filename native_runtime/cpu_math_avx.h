#pragma once
#include "host_math.h"
#include "fp8_rounding.h"
#include <immintrin.h>
#include <array>
// Translation-unit-local helpers keep AVX-512 instructions out of the AVX2 fallback.
namespace {
float half_to_float(uint16_t value) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
}
struct AvxMath : HostMath {
    // This translation unit is dispatched only after AVX/F16C and OS support
    // are checked. Inline hardware conversions avoid a scalar software encoder
    // for every published element, while preserving the reference NaN rules.
    static float round_half(float input) {
        if ((bits(input) & 0x7fffffffu) >= 0x7f800000u)
            return input;
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(input), 0)));
    }
    static uint32_t encode_half(float input) {
        const auto raw = bits(input), magnitude = raw & 0x7fffffffu;
        if (magnitude >= 0x7f800000u)
            return (raw >> 16 & 0x8000u) | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u);
        return unsigned(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(input), 0))) & 65535u;
    }
    static float decode_half(uint32_t input) {
        if ((input & 0x7c00u) == 0x7c00u)
            return value(((input & 0x8000u) << 16) | 0x7f800000u | ((input & 1023u) << 13));
        return half_to_float(uint16_t(input));
    }
    static float round_e4m3(float input) {
        return phoenix_cpp::round_e4m3_bits<HostMath>(input);
    }
    static float fma(float left, float right, float addend) {
        return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(left), _mm_set_ss(right), _mm_set_ss(addend)));
    }
};
// Decode all 256 E4M3 byte codes once. Model loading rejects NaN codes;
// this table keeps the inner GEMM loop free of scalar decoding arithmetic.
const std::array<float, 256>& fp8_table() {
    static const auto table = [] {
        std::array<float, 256> values{};
        for (unsigned code = 0; code < 256; ++code) {
            unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
            float magnitude = exponent ? std::ldexp(1.f + float(mantissa) / 8, int(exponent) - 7)
                                       : float(mantissa) / 512;
            values[code] = (code & 128) ? -magnitude : magnitude;
        }
        return values;
    }();
    return table;
}
}
