#pragma once

// Include after the selected memory variant defines MemoryRange and bounds
// checks. Variant-specific shared memory, FP8 and async-copy primitives stay in
// their own headers; only these identical 16-byte global accesses are shared.
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                       uint32_t& word_z, uint32_t& word_w,
                       const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, sizeof(uint4), false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4 words = __ldca(reinterpret_cast<const uint4*>(address));
    word_x = words.x;
    word_y = words.y;
    word_z = words.z;
    word_w = words.w;
}

template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_global_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                              uint32_t& word_z, uint32_t& word_w,
                              const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, sizeof(uint4), false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4 words = __ldcg(reinterpret_cast<const uint4*>(address));
    word_x = words.x;
    word_y = words.y;
    word_z = words.z;
    word_w = words.w;
}

template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words_streaming(uint64_t address, uint32_t word_x, uint32_t word_y,
                           uint32_t word_z, uint32_t word_w,
                           const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, sizeof(uint4), true, memory_ranges, error_counts))
        return;
    __stcs(reinterpret_cast<uint4*>(address), make_uint4(word_x, word_y, word_z, word_w));
}
