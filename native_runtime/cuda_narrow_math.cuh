#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
// Hardware conversions implement the same explicit rounding boundaries as the
// portable integer reference. Preserve its NaN payload and signed-zero rules.
struct CudaNarrowMath {
    __device__ static float round_half(float value) {
        return isfinite(value) ? __half2float(__float2half_rn(value)) : value;
    }
    __device__ static uint32_t encode_half(float value) {
        const uint32_t bits = __float_as_uint(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u)
            return ((bits >> 16u) & 0x8000u) | 0x7e00u;
        return __half_as_ushort(__float2half_rn(value));
    }
    __device__ static float decode_half(uint32_t bits) {
        if ((bits & 0x7c00u) == 0x7c00u)
            return __uint_as_float(((bits & 0x8000u) << 16u) | 0x7f800000u |
                                   ((bits & 1023u) << 13u));
        return __half2float(__ushort_as_half(uint16_t(bits)));
    }
    __device__ static float round_e4m3(float value) {
        if (isnan(value))
            return 448.f;
        const __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
        const float result = __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
        return result == 0.f ? (value < 0.f ? -0.f : 0.f) : result;
    }
};
