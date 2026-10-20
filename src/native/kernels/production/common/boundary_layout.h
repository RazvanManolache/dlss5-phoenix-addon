#pragma once
#include "word_arithmetic.h"

// One item addresses four adjacent uint32 words, not one channel scalar.
// Strides are in uint32-word units; only base_address is a byte address.
// Strides/origins retain their signed-word interpretation from the launch ABI.
struct BoundaryVectorLayout {
    uint64_t base_address;
    uint32_t component_stride_words;
    uint32_t row_stride_words;
    uint32_t column_origin;
};

#ifdef __CUDACC__
#define PHX_BOUNDARY_INLINE __host__ __device__ __forceinline__
#else
#define PHX_BOUNDARY_INLINE inline
#endif

// Map (component, row, local column) to the first word of a 16-byte vector.
// Widen each signed product separately, then add modulo 2^64. In particular,
// column_origin + column must wrap in uint32 BEFORE its signed widening.
PHX_BOUNDARY_INLINE constexpr uint64_t boundary_vector_address(
    BoundaryVectorLayout layout, uint32_t component, uint32_t row, uint32_t column) {
    const uint32_t absolute_column = layout.column_origin + column;
    const uint64_t component_offset =
        uint64_t(int64_t(int32_t(component)) * int64_t(int32_t(layout.component_stride_words)));
    const uint64_t row_offset =
        uint64_t(int64_t(int32_t(row)) * int64_t(int32_t(layout.row_stride_words)));
    constexpr uint32_t words_per_vector = 4;
    const uint64_t column_offset = uint64_t(int64_t(int32_t(absolute_column)) * words_per_vector);
    // Signed offsets are added modulo the address width, then scaled once.
    return scaled_word_address(layout.base_address, component_offset + row_offset + column_offset);
}

#undef PHX_BOUNDARY_INLINE
