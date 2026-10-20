#pragma once
#include "word_arithmetic.h"

#ifdef __CUDACC__
#define PHX_GEOMETRY_INLINE __host__ __device__ __forceinline__
#else
#define PHX_GEOMETRY_INLINE inline
#endif

struct UnsignedWordDivisor {
    uint32_t value;
    uint32_t reciprocal; // floor((2^32 - 1) / value), zero for a zero divisor.
};

// Prepare once outside coordinate loops. Integer arithmetic makes this independent
// of reciprocal approximation, FTZ behavior and float-to-integer conversion limits.
PHX_GEOMETRY_INLINE constexpr UnsignedWordDivisor make_word_divisor(uint32_t divisor) {
    return {divisor, divisor ? UINT32_MAX / divisor : 0u};
}

PHX_GEOMETRY_INLINE constexpr uint32_t exact_word_quotient(
    uint32_t numerator, UnsignedWordDivisor prepared) {
    if (!prepared.value)
        return UINT32_MAX; // Existing scheduling sentinel; never execute C++ / 0.
    const uint32_t quotient = uint32_t((uint64_t(prepared.reciprocal) * numerator) >> 32);
    const uint32_t remainder = numerator - quotient * prepared.value;
    // With M=2^32 and m=floor((M-1)/d), 0 < M/d-m <= 1. For n<M,
    // floor(n*m/M) underestimates floor(n/d) by at most one. Therefore the
    // product cannot exceed n and one remainder correction gives exact division.
    return quotient + uint32_t(remainder >= prepared.value);
}

PHX_GEOMETRY_INLINE constexpr uint32_t exact_signed_word_quotient(
    uint32_t numerator_bits, uint32_t divisor_bits) {
    if (!divisor_bits)
        return UINT32_MAX;
    uint32_t quotient = exact_word_quotient(signed_word_magnitude(numerator_bits),
                                          make_word_divisor(signed_word_magnitude(divisor_bits)));
    // Unsigned magnitudes support INT_MIN as either operand. INT_MIN/-1 has the
    // defined bit result 0x80000000, avoiding signed C++ overflow.
    return (numerator_bits ^ divisor_bits) & 0x80000000u ? 0u - quotient : quotient;
}

template <bool SignedIndices>
PHX_GEOMETRY_INLINE constexpr uint32_t geometry_tail_steps(
    uint32_t next_index, uint32_t end_index, uint32_t thread_stride) {
    const bool has_next = SignedIndices ? int32_t(next_index) < int32_t(end_index)
                                       : next_index < end_index;
    const uint32_t remaining_minus_one = has_next ? end_index - next_index - 1u : 0u;
    return uint32_t(has_next) + exact_word_quotient(remaining_minus_one,
                                                  make_word_divisor(thread_stride));
}

#undef PHX_GEOMETRY_INLINE
