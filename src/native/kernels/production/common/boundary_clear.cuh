#pragma once
#include "boundary_layout.h"
#include "boundary_schedule.cuh"

// Components are 4, 8, 16, 32 or 64 vectors per spatial coordinate. The flat
// work index interleaves component first, then column, then row.
template <unsigned ComponentBits>
__device__ __forceinline__ uint64_t boundary_item_address(
    BoundaryVectorLayout layout, UnsignedWordDivisor width, uint32_t index) {
    static_assert(ComponentBits >= 2 && ComponentBits <= 6, "Production boundary vector layout");
    const uint32_t spatial_index = index >> ComponentBits;
    const uint32_t row = prepared_unsigned_word_quotient(spatial_index, width);
    const uint32_t column = spatial_index - width.value * row;
    const uint32_t component = index & ((1u << ComponentBits) - 1u);
    return boundary_vector_address(layout, component, row, column);
}

// StoreWord is a compile-time callable supplied by the caller. Input preparation
// keeps its checked PTX stores; window kernels keep their volatile pipeline
// stores. No function pointer, virtual dispatch or new memory primitive is used.
template <class StoreWord>
__device__ __forceinline__ void clear_boundary_vector(uint64_t address, StoreWord store_word) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word)
        store_word(address + word * sizeof(uint32_t));
}

// Preconditions: the caller has checked width > 0 and current index < extent;
// thread_stride is the nonzero block thread count. Extents and increments must
// stay in the supported signed-positive index geometry. Storage remains uint32;
// the final loop preserves its signed-index comparison. No barriers move.
template <unsigned ComponentBits, class StoreWord>
__device__ __forceinline__ void clear_boundary_rows(
    BoundaryVectorLayout layout, uint32_t width, uint32_t index, uint32_t extent,
    uint32_t thread_stride, StoreWord store_word) {
    const uint32_t further_steps = boundary_tail_steps<true>(index + thread_stride, extent, thread_stride);
    const UnsignedWordDivisor prepared_width = prepare_unsigned_word_divisor(width);

    // Preserve the original 0..3-item prefix followed by four-item groups.
    // This is not a change to the neural network or the amount of padding.
    for (uint32_t tail = (further_steps + 1u) & 3u; tail; --tail) {
        clear_boundary_vector(boundary_item_address<ComponentBits>(layout, prepared_width, index), store_word);
        index += thread_stride;
    }
    if (further_steps < 3u)
        return;

    do {
        // Compute all four addresses before publishing any vector, retaining
        // independent quotient work and ascending per-thread store order.
        uint64_t addresses[4];
#pragma unroll
        for (unsigned item = 0; item < 4; ++item)
            addresses[item] = boundary_item_address<ComponentBits>(
                layout, prepared_width, index + item * thread_stride);
#pragma unroll
        for (unsigned item = 0; item < 4; ++item)
            clear_boundary_vector(addresses[item], store_word);
        index += 4u * thread_stride;
    } while (int32_t(index) < int32_t(extent));
}
