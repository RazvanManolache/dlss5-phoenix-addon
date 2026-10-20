#pragma once
#include <cstdint>

// These bit contracts are shared with CPU regression tests. Keep unsigned
// arithmetic: INT_MIN has a representable unsigned magnitude, not a signed one.
#ifdef __CUDACC__
#define PHX_WORD_INLINE __host__ __device__ __forceinline__
#else
#define PHX_WORD_INLINE inline
#endif

PHX_WORD_INLINE constexpr uint32_t signed_word_magnitude(uint32_t signed_bits) {
    return int32_t(signed_bits) < 0 ? 0u - signed_bits : signed_bits;
}

// Signed division truncates toward zero. A bare arithmetic right shift rounds
// negative values down instead. Return bits so signed INT_MIN remains defined.
template <unsigned Shift>
PHX_WORD_INLINE constexpr uint32_t truncate_signed_power_of_two(uint32_t signed_bits) {
    static_assert(Shift < 31, "Positive signed divisor must fit int32");
    return uint32_t(int32_t(signed_bits) / (int32_t(1) << Shift));
}

PHX_WORD_INLINE constexpr uint64_t combine_words(uint32_t low_word, uint32_t high_word) {
    return uint64_t(low_word) | (uint64_t(high_word) << 32);
}

PHX_WORD_INLINE void split_address_words(uint32_t& low_word, uint32_t& high_word,
                                        uint64_t address) {
    low_word = uint32_t(address);
    high_word = uint32_t(address >> 32);
}

// Offsets count uint32 words. Scaling is performed once at full address width;
// the shifted offset already contains the carry out of its low 32-bit word.
// Adding that carry again skips 4 GiB for large or negative signed offsets.
// The unsigned operations also define wrap at the 64-bit address boundary.
PHX_WORD_INLINE constexpr uint64_t scaled_word_address(uint64_t base_address,
                                                       uint64_t word_offset) {
    return base_address + word_offset * uint64_t(sizeof(uint32_t));
}

#undef PHX_WORD_INLINE
