// Frame-boundary scalar/vector accesses. Preserve the PTX memory scope and ordering.
// Bounds checks compile out of the production entry point.

#include "async_memory.cuh"

template <bool CheckBounds>
__device__ __forceinline__ bool check_boundary_address(uint64_t address, uint32_t byte_count,
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
        if (!valid || address & byte_count - 1) {
            atomicAdd(error_counts + unsigned(write), 1u);
            return false;
        }
    }
    return true;
}
template <bool CheckBounds>
__device__ __forceinline__ uint32_t read_boundary_word(uint64_t address,
                                                       const MemoryRange* memory_ranges,
                                                       uint32_t* error_counts) {
    if (!check_boundary_address<CheckBounds>(address, 4, false, memory_ranges, error_counts))
        return 0;
    uint32_t loaded_word;
    asm volatile("ld.u32 %0,[%1];" : "=r"(loaded_word) : "l"(address) : "memory");
    return loaded_word;
}
template <bool CheckBounds>
__device__ __forceinline__ uint32_t read_boundary_half(uint64_t address,
                                                       const MemoryRange* memory_ranges,
                                                       uint32_t* error_counts) {
    if (!check_boundary_address<CheckBounds>(address, 2, false, memory_ranges, error_counts))
        return 0;
    uint16_t loaded_word;
    asm volatile("ld.global.u16 %0,[%1];" : "=h"(loaded_word) : "l"(address) : "memory");
    return loaded_word;
}
template <bool CheckBounds>
__device__ __forceinline__ void write_boundary_word(uint64_t address, uint32_t loaded_word,
                                                    const MemoryRange* memory_ranges,
                                                    uint32_t* error_counts) {
    if (!check_boundary_address<CheckBounds>(address, 4, true, memory_ranges, error_counts))
        return;
    asm volatile("st.global.u32 [%0],%1;" ::"l"(address), "r"(loaded_word) : "memory");
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_two_boundary_words(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                        const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_boundary_address<CheckBounds>(address, 8, false, memory_ranges, error_counts)) {
        word_x = word_y = 0;
        return;
    }
    asm volatile("ld.relaxed.cta.global.v2.u32 {%0,%1},[%2];"
                 : "=r"(word_x), "=r"(word_y)
                 : "l"(address)
                 : "memory");
}
template <bool CheckBounds, int MemoryScope>
__device__ __forceinline__ void
load_four_boundary_words(uint64_t address, uint32_t& word_x, uint32_t& word_y, uint32_t& word_z,
                         uint32_t& word_w, const MemoryRange* memory_ranges,
                         uint32_t* error_counts) {
    if (!check_boundary_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    if constexpr (MemoryScope == 0)
        asm volatile("ld.relaxed.cta.global.v4.u32 {%0,%1,%2,%3},[%4];"
                     : "=r"(word_x), "=r"(word_y), "=r"(word_z), "=r"(word_w)
                     : "l"(address)
                     : "memory");
    else
        asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3},[%4];"
                     : "=r"(word_x), "=r"(word_y), "=r"(word_z), "=r"(word_w)
                     : "l"(address)
                     : "memory");
}
