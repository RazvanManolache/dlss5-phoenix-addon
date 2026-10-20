#pragma once
#include "integer_geometry.h"

// CUDA entry points share the integer implementation exercised by CPU tests.
__device__ __forceinline__ UnsignedWordDivisor prepare_unsigned_word_divisor(uint32_t divisor) {
    return make_word_divisor(divisor);
}

__device__ __forceinline__ uint32_t prepared_unsigned_word_quotient(
    uint32_t numerator, UnsignedWordDivisor prepared) {
    return exact_word_quotient(numerator, prepared);
}

__device__ __forceinline__ uint32_t unsigned_word_quotient(uint32_t numerator,
                                                        uint32_t divisor) {
    return prepared_unsigned_word_quotient(numerator, prepare_unsigned_word_divisor(divisor));
}
