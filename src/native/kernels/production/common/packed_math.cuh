// Packed word/address access and guarded global/shared memory operations.
// Bounds checks are compiled out for production launches; word offsets remain byte offsets until indexing.

#include "window_math.cuh"
#include "word_arithmetic.h"

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
