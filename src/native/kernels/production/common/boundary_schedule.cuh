#pragma once
#include "unsigned_division.cuh"

// The caller already owns one valid item. Count only the further strided items
// starting at next_index; the existing epilogue separately adds its current item
// and divides the work into a 0..3-item tail and four-item unrolled groups.
// Keep signed and unsigned bounds explicit: flattened extents used signed-word
// comparisons, whereas the fixed-size boundary tile used unsigned comparisons.
template <bool SignedIndices>
__device__ __forceinline__ uint32_t boundary_tail_steps(uint32_t next_index,
                                                      uint32_t end_index,
                                                      uint32_t thread_stride) {
    return geometry_tail_steps<SignedIndices>(next_index, end_index, thread_stride);
}
