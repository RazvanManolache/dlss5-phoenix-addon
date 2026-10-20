#pragma once
#include "integer_geometry.h"

// Truncation toward zero for every int32 bit pattern, including INT_MIN divisors.
// Zero and INT_MIN/-1 retain their explicit unsigned-bit results.
__device__ __forceinline__ uint32_t signed_tile_quotient(uint32_t numerator_bits,
                                                       uint32_t divisor_bits) {
    return exact_signed_word_quotient(numerator_bits, divisor_bits);
}
