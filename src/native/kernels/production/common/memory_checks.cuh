#pragma once

// Shared bounds/alignment policy. Selected memory variants supply MemoryRange;
// CheckBounds=false removes the entire diagnostic path from production code.

template <bool CheckBounds>
__device__ __forceinline__ bool check_vector_address(uint64_t address, uint32_t byte_count,
                                                     bool write, const MemoryRange* memory_ranges,
                                                     uint32_t* error_counts) {
    if constexpr (CheckBounds) {
        bool valid = false;
        for (int range_index = 0; range_index < 16; ++range_index)
            valid |= address >= memory_ranges[range_index].base &&
                     address - memory_ranges[range_index].base <= memory_ranges[range_index].size &&
                     byte_count <= memory_ranges[range_index].size -
                                       (address - memory_ranges[range_index].base) &&
                     (!write || memory_ranges[range_index].writable);
        if (!valid || address & 15) {
            atomicAdd(error_counts + unsigned(write), 1u);
            return false;
        }
    }
    return true;
}
