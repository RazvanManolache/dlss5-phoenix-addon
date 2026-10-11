// Specialized native memory-access helper for the selected kernel variant.
// Vector width, address space and half/FP8 packing distinguish this helper from its siblings.
// Do not merge variants solely because their function signatures match.

#include "async_memory_native_global_fp8.cuh"
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_cached(uint64_t address, uint32_t& address_value, uint32_t& word_y,
                       uint32_t& word_z, uint32_t& weights, const MemoryRange* memory_ranges,
                       uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        address_value = word_y = word_z = weights = 0;
        return;
    }
    const uint4 result_word = __ldca(reinterpret_cast<const uint4*>(address));
    address_value = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    weights = result_word.w;
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
    const uint4 result_word = __ldcg(reinterpret_cast<const uint4*>(address));
    address_value = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    weights = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words_streaming(uint64_t address, uint32_t address_value, uint32_t word_y,
                           uint32_t word_z, uint32_t weights, const MemoryRange* memory_ranges,
                           uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    __stcs(reinterpret_cast<uint4*>(address), make_uint4(address_value, word_y, word_z, weights));
}
