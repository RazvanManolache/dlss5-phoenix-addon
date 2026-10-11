// Packed word/address access and guarded global/shared memory operations.
// Bounds checks are compiled out for production launches; word offsets remain byte offsets until indexing.

#include "window_math.cuh"
__device__ __forceinline__ uint64_t combine_words(uint32_t low_word, uint32_t high_word) {
    return uint64_t(low_word) | (uint64_t(high_word) << 32);
}
__device__ __forceinline__ void split_address_words(uint32_t& low_word, uint32_t& high_word,
                                                    uint64_t address_value) {
    low_word = uint32_t(address_value);
    high_word = uint32_t(address_value >> 32);
}

// Convert a packed word offset to a byte address. Normal tensor offsets use
// ordinary 64-bit addition. Existing kernels also increment the high word
// when scaling an oversized low offset overflows, even if adding the
// wrapped low part to the base does not. Preserve that edge behavior explicitly;
// silently replacing it with pointer addition would change offsets >= 2^30 words.
__device__ __forceinline__ uint64_t scaled_word_address(uint64_t base_address,
                                                        uint64_t word_offset) {
    const uint64_t byte_address = base_address + (word_offset << 2);
    const bool scaled_low_overflow = uint32_t(word_offset) >= (uint32_t(1) << 30);
    const bool wrapped_low_addition = uint32_t(byte_address) < uint32_t(base_address);
    return byte_address + (uint64_t(scaled_low_overflow && !wrapped_low_addition) << 32);
}

__device__ __forceinline__ uint32_t permute_signed(uint32_t operand_a, uint32_t operand_b,
                                                   uint32_t selector) {
    uint32_t destination_words;
    asm("prmt.b32 %0,%1,%2,%3;"
        : "=r"(destination_words)
        : "r"(operand_a), "r"(operand_b), "r"(selector));
    return destination_words;
}

template <bool CheckBounds>
__device__ __forceinline__ uint32_t global_read(uint64_t address, const uint8_t* input,
                                                const uint8_t* weights, uint32_t input_byte_count,
                                                uint32_t weight_byte_count,
                                                uint32_t* error_counts) {
    if (CheckBounds &&
        !(address >= uint64_t(input) && address + 4 <= uint64_t(input) + input_byte_count ||
          address >= uint64_t(weights) && address + 4 <= uint64_t(weights) + weight_byte_count)) {
        atomicAdd(error_counts, 1);
        return 0;
    }
    return load_word(reinterpret_cast<const void*>(address));
}
template <bool CheckBounds>
__device__ __forceinline__ void global_write(uint64_t address, uint32_t result_word,
                                             uint8_t* output, uint32_t byte_count,
                                             uint32_t* error_counts) {
    if (CheckBounds &&
        !(address >= uint64_t(output) && address + 4 <= uint64_t(output) + byte_count)) {
        atomicAdd(error_counts + 1, 1);
        return;
    }
    store_word(reinterpret_cast<void*>(address), result_word);
}
template <bool CheckBounds>
__device__ __forceinline__ uint32_t shared_read(uint32_t* shared_tile, uint32_t shared_offset,
                                                uint32_t byte_count, uint32_t* error_counts) {
    if (CheckBounds && !(shared_offset + 4 <= byte_count && shared_offset % 4 == 0)) {
        atomicAdd(error_counts + 2, 1);
        return 0;
    }
    return shared_tile[shared_offset / 4];
}
template <bool CheckBounds>
__device__ __forceinline__ void shared_write(uint32_t* shared_tile, uint32_t shared_offset,
                                             uint32_t result_word, uint32_t byte_count,
                                             uint32_t* error_counts) {
    if (CheckBounds && !(shared_offset + 4 <= byte_count && shared_offset % 4 == 0)) {
        atomicAdd(error_counts + 3, 1);
        return;
    }
    shared_tile[shared_offset / 4] = result_word;
}
