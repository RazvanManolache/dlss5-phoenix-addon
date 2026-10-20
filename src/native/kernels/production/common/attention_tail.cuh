#pragma once

// Include after the kernel's existing half-pair primitives. Tail count is an
// unsigned integer; the correction is converted to half BEFORE subtraction
// from each half lane. Keep those rounding boundaries and the FP32 FTZ multiply.
__device__ __forceinline__ uint32_t subtract_attention_tail(uint32_t sum_pair,
                                                           uint32_t tail_count) {
    // Existing finite constant chain: clamp half(0x3ed6) to [0x3dc2,0x3fe9],
    // then reinterpret ((bits << 4) + 0x4000) & 0xffff as half. The clamp leaves
    // 0x3ed6 unchanged, producing half(0x2d60), exactly 43/512 in FP32.
    constexpr float coefficient = 43.f / 512.f;
    const float count = __uint2float_rn(tail_count);
    float correction;
    asm("mul.rn.ftz.f32 %0,%1,%2;" : "=f"(correction) : "f"(coefficient), "f"(count));
    const uint32_t correction_pair = pack_half(0u, __float_as_uint(correction));
    return add_half_pairs(sum_pair, select_half_operand<0, false, true>(correction_pair));
}
