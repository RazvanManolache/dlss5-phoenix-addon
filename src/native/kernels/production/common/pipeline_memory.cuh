// Native pipeline loads/stores and synchronization primitives.
// Ordering, cache modes and async completion tokens are part of the producer/consumer contract.

#include "packed_math.cuh"
struct MemoryRange {
    uint64_t base, size, writable;
};
__device__ __forceinline__ bool
is_word_address_allowed(uint64_t address, const MemoryRange* memory_ranges, bool write) {
    for (int range_index = 0; range_index < 16; range_index++)
        if (address >= memory_ranges[range_index].base &&
            address + 4 <= memory_ranges[range_index].base + memory_ranges[range_index].size &&
            (!write || memory_ranges[range_index].writable))
            return true;
    return false;
}
template <bool CheckBounds>
__device__ __forceinline__ uint32_t pipeline_read(uint64_t address,
                                                  const MemoryRange* memory_ranges,
                                                  uint32_t* error_counts) {
    if (CheckBounds && !is_word_address_allowed(address, memory_ranges, false)) {
        atomicAdd(error_counts, 1);
        return 0;
    }
    return *reinterpret_cast<volatile const uint32_t*>(address);
}
template <bool CheckBounds>
__device__ __forceinline__ void pipeline_write(uint64_t address, uint32_t result_word,
                                               const MemoryRange* memory_ranges,
                                               uint32_t* error_counts) {
    if (CheckBounds && !is_word_address_allowed(address, memory_ranges, true)) {
        atomicAdd(error_counts + 1, 1);
        return;
    }
    *reinterpret_cast<volatile uint32_t*>(address) = result_word;
}
template <bool CheckBounds>
__device__ __forceinline__ void pipeline_add_half(uint64_t address, uint32_t value,
                                                  const MemoryRange* memory_ranges,
                                                  uint32_t* error_counts) {
    if (CheckBounds && !is_word_address_allowed(address, memory_ranges, true)) {
        atomicAdd(error_counts + 1, 1);
        return;
    }
    uint32_t ignored;
    asm volatile("atom.add.noftz.f16x2 %0,[%1],%2;"
                 : "=r"(ignored)
                 : "l"(address), "r"(value)
                 : "memory");
}
template <bool CheckBounds>
__device__ __forceinline__ void bulk_copy(uint32_t* shared_tile, uint32_t destination_offset,
                                          uint64_t source_address, uint32_t byte_count,
                                          uint32_t capacity, const MemoryRange* memory_ranges,
                                          uint32_t* error_counts) {
    for (uint32_t byte_offset = 0; byte_offset < byte_count; byte_offset += 4)
        shared_write<CheckBounds>(
            shared_tile, destination_offset + byte_offset,
            pipeline_read<CheckBounds>(source_address + byte_offset, memory_ranges, error_counts),
            capacity, error_counts);
}
