// Aligned 16-byte feature/weight access. One vector instruction replaces four
// scalar requests while preserving the original cache modes and async helpers.
// Callers already satisfy check_vector_address's 16-byte alignment contract.
#include "async_memory.cuh"

template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                       uint32_t& word_z, uint32_t& word_w,
                       const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4 packed_words = __ldca(reinterpret_cast<const uint4*>(address));
    word_x = packed_words.x;
    word_y = packed_words.y;
    word_z = packed_words.z;
    word_w = packed_words.w;
}

template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_global_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                              uint32_t& word_z, uint32_t& word_w,
                              const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4 packed_words = __ldcg(reinterpret_cast<const uint4*>(address));
    word_x = packed_words.x;
    word_y = packed_words.y;
    word_z = packed_words.z;
    word_w = packed_words.w;
}

template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words_streaming(uint64_t address, uint32_t word_x, uint32_t word_y,
                           uint32_t word_z, uint32_t word_w,
                           const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    __stcs(reinterpret_cast<uint4*>(address), make_uint4(word_x, word_y, word_z, word_w));
}
