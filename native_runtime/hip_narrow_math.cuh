#pragma once
#include <hip/hip_fp16.h>
#include "fp8_rounding.h"

// Native half instructions retain the explicit publication boundaries. E4M3
// rounding stays integer-based on GPUs without native FP8 conversion support.
struct HipNarrowMath {
    __device__ static uint32_t bits(float input) {
        return __float_as_uint(input);
    }
    __device__ static float value(uint32_t input) {
        return __uint_as_float(input);
    }
    __device__ static float round_half(float input) {
        return (bits(input) & 0x7fffffffu) >= 0x7f800000u ? input
                                                          : __half2float(__float2half_rn(input));
    }
    __device__ static uint32_t encode_half(float input) {
        const auto raw = bits(input), magnitude = raw & 0x7fffffffu;
        if (magnitude > 0x7f800000u)
            return (raw >> 16 & 0x8000u) | 0x7e00u;
        return __half_as_ushort(__float2half_rn(input));
    }
    __device__ static float decode_half(uint32_t input) {
        if ((input & 0x7c00u) == 0x7c00u)
            return value(((input & 0x8000u) << 16) | 0x7f800000u | ((input & 1023u) << 13));
        return __half2float(__ushort_as_half(uint16_t(input)));
    }
    __device__ static float round_e4m3(float input) {
        return phoenix_cpp::round_e4m3_bits<HipNarrowMath>(input);
    }
};
