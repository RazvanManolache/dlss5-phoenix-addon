// Specialized native memory-access helper for the selected kernel variant.
// Vector width, address space and half/FP8 packing distinguish this helper from its siblings.
// Do not merge variants solely because their function signatures match.

#include "async_memory.cuh"
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_cached(uint64_t address, uint32_t& address_value, uint32_t& word_y,
                       uint32_t& word_z, uint32_t& weights, const MemoryRange* memory_ranges,
                       uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        address_value = word_y = word_z = weights = 0;
        return;
    }
#if PHOENIX_STYLE_ARCH >= 890
    // These callers already require 16-byte alignment. One packed access
    // avoids four scalar address/load operations on the native FP8 paths.
    const uint4 packed = __ldca(reinterpret_cast<const uint4*>(address));
    address_value = packed.x; word_y = packed.y; word_z = packed.z; weights = packed.w;
#else
    const uint32_t* word_pointer = reinterpret_cast<const uint32_t*>(address);
    address_value = __ldca(word_pointer);
    word_y = __ldca(word_pointer + 1);
    word_z = __ldca(word_pointer + 2);
    weights = __ldca(word_pointer + 3);
#endif
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_global_cached(uint64_t address, uint32_t& address_value, uint32_t& word_y,
                              uint32_t& word_z, uint32_t& weights, const MemoryRange* memory_ranges,
                              uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        address_value = word_y = word_z = weights = 0;
        return;
    }
#if PHOENIX_STYLE_ARCH >= 890
    const uint4 packed = __ldcg(reinterpret_cast<const uint4*>(address));
    address_value = packed.x; word_y = packed.y; word_z = packed.z; weights = packed.w;
#else
    const uint32_t* word_pointer = reinterpret_cast<const uint32_t*>(address);
    address_value = __ldcg(word_pointer);
    word_y = __ldcg(word_pointer + 1);
    word_z = __ldcg(word_pointer + 2);
    weights = __ldcg(word_pointer + 3);
#endif
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words_streaming(uint64_t address, uint32_t address_value, uint32_t word_y,
                           uint32_t word_z, uint32_t weights, const MemoryRange* memory_ranges,
                           uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    uint32_t* word_pointer = reinterpret_cast<uint32_t*>(address);
    __stcs(word_pointer, address_value);
    __stcs(word_pointer + 1, word_y);
    __stcs(word_pointer + 2, word_z);
    __stcs(word_pointer + 3, weights);
}
