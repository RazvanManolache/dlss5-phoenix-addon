#pragma once
// Compile-time compatibility helpers. The sm_120 instruction paths remain in
// their original helpers; no architecture branch is evaluated during a frame.
// FP8 -> paired FP16 MMA decomposition follows the compatibility approach in
// OpenNR (Carlos Lopez Jr., MIT); see third_party/opennr/LICENSE and THIRD_PARTY.md.
#include <stdint.h>
#include <cuda_fp16.h>

namespace phoenix_architecture {
__device__ __forceinline__ void copy_shared_word(uint32_t destination, uint64_t source) {
    uint32_t word;
    asm volatile("ld.global.ca.u32 %0,[%1];" : "=r"(word) : "l"(source) : "memory");
    asm volatile("st.shared.u32 [%0],%1;" :: "r"(destination), "r"(word) : "memory");
}
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 890 || defined(PHOENIX_ARCHITECTURE_TEST)
// Integer conversions avoid expanding the general CUDA conversion routines into
// every matrix fragment on devices without FP8 instructions. Round to nearest,
// ties to even; preserve signed zero and saturate infinities to finite E4M3.
__host__ __device__ __forceinline__ uint8_t encode_fp8_value(uint16_t half) {
    const unsigned magnitude = half & 0x7fff, sign = (half >> 8) & 0x80;
    if (magnitude > 0x7c00) return 0x7f;
    if (magnitude >= 0x5f00) return uint8_t(sign | 126);
    if (magnitude >= 0x2400)
        return uint8_t(sign | ((magnitude - 0x2000 + 63 + ((magnitude >> 7) & 1)) >> 7));
    if (magnitude <= 0x1400) return uint8_t(sign);
    const unsigned shift = 16 - (magnitude >> 10);
    const unsigned significand = 0x400 | (magnitude & 0x3ff);
    const unsigned rounded = (significand + (1u << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
    return uint8_t(sign | rounded);
}
__host__ __device__ __forceinline__ uint16_t decode_fp8_value(uint8_t code) {
    const unsigned magnitude = code & 127, sign = (code & 128) << 8;
    if (magnitude == 127) return uint16_t(sign | 0x7e00);
    const unsigned exponent = magnitude >> 3, fraction = magnitude & 7;
    if (exponent) return uint16_t(sign | ((exponent + 8) << 10) | (fraction << 7));
    if (!fraction) return uint16_t(sign);
    const unsigned shift = fraction < 2 ? 2 : fraction < 4 ? 1 : 0;
    return uint16_t(sign | ((8 - shift) << 10) | ((fraction << (8 + shift)) & 0x3ff));
}
__host__ __device__ __forceinline__ uint16_t encode_fp8_pair(uint32_t halves) {
    return uint16_t(encode_fp8_value(uint16_t(halves))) |
           (uint16_t(encode_fp8_value(uint16_t(halves >> 16))) << 8);
}
__host__ __device__ __forceinline__ uint32_t decode_fp8_pair(uint16_t bytes) {
    return uint32_t(decode_fp8_value(uint8_t(bytes))) |
           (uint32_t(decode_fp8_value(uint8_t(bytes >> 8))) << 16);
}
// Use OpenNR's paired-k16 grouping: bytes 0/2 become one FP16 pair, bytes 1/3
// the other. Both operands must use the same grouping of reduction channels.
__host__ __device__ __forceinline__ void decode_fp8_fragment(uint32_t word, uint32_t& even, uint32_t& odd) {
    even = decode_fp8_pair(uint16_t((word & 0xff) | ((word >> 8) & 0xff00)));
    odd = decode_fp8_pair(uint16_t(((word >> 8) & 0xff) | ((word >> 16) & 0xff00)));
}
// Fragment order follows mma.m16n8k16. Turing consumes two k8 steps;
// Ampere consumes one k16 step. Accumulators remain packed FP16.
__device__ __forceinline__ void half_matrix(
    uint32_t& d0, uint32_t& d1, uint32_t a0, uint32_t a1, uint32_t a2,
    uint32_t a3, uint32_t b0, uint32_t b1, uint32_t c0, uint32_t c1) {
#if __CUDA_ARCH__ < 800 || defined(PHOENIX_TEST_TURING_MMA)
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3},{%6},{%8,%9};\n"
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1},{%4,%5},{%7},{%0,%1};"
        : "=&r"(d0), "=&r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
#endif
}
__device__ __forceinline__ void fp8_matrix(
    uint32_t& d0, uint32_t& d1, uint32_t a0, uint32_t a1, uint32_t a2,
    uint32_t a3, uint32_t b0, uint32_t b1, uint32_t c0, uint32_t c1) {
    const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1};
    uint32_t low[4], high[4], right_low[2], right_high[2];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        decode_fp8_fragment(a[i], low[i], high[i]);
    }
#pragma unroll
    for (unsigned i = 0; i < 2; ++i) {
        decode_fp8_fragment(b[i], right_low[i], right_high[i]);
    }
    half_matrix(d0, d1, low[0], low[1], high[0], high[1], right_low[0], right_high[0], c0, c1);
    half_matrix(d0, d1, low[2], low[3], high[2], high[3], right_low[1], right_high[1], d0, d1);
}
#endif
} // namespace phoenix_architecture
