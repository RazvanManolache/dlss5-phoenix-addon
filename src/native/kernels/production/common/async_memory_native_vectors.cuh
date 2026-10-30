#include "architecture_policy.cuh"
// Specialized native memory-access helper for the selected kernel variant.
// Vector width, address space and half/FP8 packing distinguish this helper from its siblings.
// Do not merge variants solely because their function signatures match.

#include "memory_checks.cuh"
#include "warp_collectives.cuh"
#include "copy_barriers.cuh"

template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words(uint64_t address, uint32_t& address_value, uint32_t& word_y, uint32_t& word_z,
                uint32_t& weights, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        address_value = word_y = word_z = weights = 0;
        return;
    }
    const uint4 result_word = *reinterpret_cast<const uint4*>(address);
    address_value = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    weights = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words(uint64_t address, uint32_t address_value, uint32_t word_y, uint32_t word_z,
                 uint32_t weights, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    *reinterpret_cast<uint4*>(address) = make_uint4(address_value, word_y, word_z, weights);
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_shared_words(uint32_t* shared_words, uint32_t shared_offset, uint32_t& address_value,
                       uint32_t& word_y, uint32_t& word_z, uint32_t& weights,
                       uint32_t capacity_bytes, const MemoryRange*, uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (shared_offset > capacity_bytes || 16 > capacity_bytes - shared_offset ||
            shared_offset & 15) {
            atomicAdd(error_counts, 1u);
            address_value = word_y = word_z = weights = 0;
            return;
        }
    const uint4 result_word = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const unsigned char*>(shared_words) + shared_offset);
    address_value = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    weights = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_shared_words(uint32_t* shared_words, uint32_t shared_offset, uint32_t address_value,
                        uint32_t word_y, uint32_t word_z, uint32_t weights, uint32_t capacity_bytes,
                        const MemoryRange*, uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (shared_offset > capacity_bytes || 16 > capacity_bytes - shared_offset ||
            shared_offset & 15) {
            atomicAdd(error_counts, 1u);
            return;
        }
    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(shared_words) + shared_offset) =
        make_uint4(address_value, word_y, word_z, weights);
}

template <bool CheckBounds>
__device__ __forceinline__ void check_single_lane(uint32_t* error_counts) {
    if constexpr (CheckBounds) {
        uint32_t mask;
        asm volatile("activemask.b32 %0;" : "=r"(mask));
        if (!mask || (mask & (mask - 1)))
            atomicAdd(error_counts + 2, 1u);
    }
}

__device__ __forceinline__ uint32_t cosine_radians(uint32_t operand_a) {
    float result_word;
    asm("cos.approx.ftz.f32 %0,%1;" : "=f"(result_word) : "f"(__uint_as_float(operand_a)));
    return __float_as_uint(result_word);
}
__device__ __forceinline__ uint32_t sine_radians(uint32_t operand_a) {
    float result_word;
    asm("sin.approx.ftz.f32 %0,%1;" : "=f"(result_word) : "f"(__uint_as_float(operand_a)));
    return __float_as_uint(result_word);
}

template <bool CheckBounds>
__device__ __forceinline__ uint32_t load_aligned_word(uint64_t address,
                                                      const MemoryRange* memory_ranges,
                                                      uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, false) || address & 3) {
            atomicAdd(error_counts, 1u);
            return 0;
        }
    uint32_t value;
    asm volatile("ld.global.u32 %0,[%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

template <int Lane, bool Absolute, bool Negative>
__device__ __forceinline__ uint32_t select_half_operand(uint32_t value) {
    if constexpr (Lane < 2) {
        uint32_t packed;
        if constexpr (Lane == 0)
            asm("{ .reg .b16 lo,hi; mov.b32 {lo,hi},%1; mov.b32 %0,{lo,lo}; }"
                : "=r"(packed)
                : "r"(value));
        else
            asm("{ .reg .b16 lo,hi; mov.b32 {lo,hi},%1; mov.b32 %0,{hi,hi}; }"
                : "=r"(packed)
                : "r"(value));
        value = packed;
    }
    if constexpr (Absolute)
        asm("abs.f16x2 %0,%1;" : "=r"(value) : "r"(value));
    if constexpr (Negative)
        asm("neg.f16x2 %0,%1;" : "=r"(value) : "r"(value));
    return value;
}

__device__ __forceinline__ uint32_t pack_fp8_pair(uint32_t operand_a, uint32_t operand_c) {
#if PHOENIX_STYLE_ARCH < 890
    return uint32_t(phoenix_architecture::encode_fp8_pair(operand_a)) | (operand_c << 16);
#else
    uint32_t result;
    asm("{ .reg .b16 lo,hi,unused; mov.b32 {hi,unused},%2; cvt.rn.satfinite.e4m3x2.f16x2 lo,%1; "
        "mov.b32 %0,{lo,hi}; }"
        : "=r"(result)
        : "r"(operand_a), "r"(operand_c));
    return result;
#endif
}
