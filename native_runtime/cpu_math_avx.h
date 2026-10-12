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
