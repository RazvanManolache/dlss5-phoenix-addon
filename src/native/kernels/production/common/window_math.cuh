#include "architecture_policy.cuh"
// Half conversion, pair packing and warp-matrix permutation primitives.
// Integer return values carry floating-point bits; reinterpretation is intentional, not numeric conversion.

#include "feed_forward_math.cuh"
__device__ __forceinline__ uint32_t half_to_float_bits(uint32_t half_bits) {
    float float_value;
    asm("cvt.f32.f16 %0,%1;" : "=f"(float_value) : "h"(uint16_t(half_bits)));
    return __float_as_uint(float_value);
}
// PTX puts the second converted value in the low half and the first in the
// upper half. Argument order therefore matters even though both are FP32.
__device__ __forceinline__ uint32_t pack_half(uint32_t upper_float_bits,
                                              uint32_t lower_float_bits) {
    uint32_t half_pair;
#if PHOENIX_STYLE_ARCH < 800
    asm("{ .reg .b16 lo,hi; cvt.rn.f16.f32 hi,%1; cvt.rn.f16.f32 lo,%2; mov.b32 %0,{lo,hi}; }"
        : "=r"(half_pair)
        : "f"(__uint_as_float(upper_float_bits)), "f"(__uint_as_float(lower_float_bits)));
#else
    asm("cvt.rn.f16x2.f32 %0,%1,%2;"
        : "=r"(half_pair)
        : "f"(__uint_as_float(upper_float_bits)), "f"(__uint_as_float(lower_float_bits)));
#endif
    return half_pair;
}
// Keep the hardware approximation and flush-to-zero behavior used by native
// normalization; a library sqrt/division sequence has different rounding.
__device__ __forceinline__ uint32_t rsqrt_bits(uint32_t float_bits) {
    float inverse_root;
    asm("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(inverse_root) : "f"(__uint_as_float(float_bits)));
    return __float_as_uint(inverse_root);
}
__device__ __forceinline__ uint32_t reciprocal_bits(uint32_t float_bits) {
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0,%1;" : "=f"(reciprocal) : "f"(__uint_as_float(float_bits)));
    return __float_as_uint(reciprocal);
}
// Transpose an 8x8 half tile collectively across the active warp.
__device__ __forceinline__ uint32_t transpose_half_tile(uint32_t lane_half_pair) {
    uint32_t transposed_pair;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0,%1;"
        : "=r"(transposed_pair)
        : "r"(lane_half_pair));
    return transposed_pair;
}
