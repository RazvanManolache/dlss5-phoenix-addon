// Packed feed-forward activation and publication arithmetic.
// Inline instructions preserve half-pair rounding and lane selection used by fused native kernels.

#include <stdint.h>
#include "architecture_math.cuh"

__device__ __forceinline__ uint32_t load_word(const void* word_pointer) {
    return *reinterpret_cast<const uint32_t*>(word_pointer);
}
__device__ __forceinline__ void store_word(void* word_pointer, uint32_t result_word) {
    *reinterpret_cast<uint32_t*>(word_pointer) = result_word;
}
// Each uint32_t contains two FP16 values. The two products round independently
// to FP16, so replacing this with FP32 multiplication would change inference.
__device__ __forceinline__ uint32_t multiply_half_pairs(uint32_t left_pair, uint32_t right_pair) {
    uint32_t product_pair;
    asm("mul.rn.f16x2 %0,%1,%2;" : "=r"(product_pair) : "r"(left_pair), "r"(right_pair));
    return product_pair;
}
__device__ __forceinline__ uint32_t add_half_pairs(uint32_t left_pair, uint32_t right_pair) {
    uint32_t sum_pair;
    asm("add.rn.f16x2 %0,%1,%2;" : "=r"(sum_pair) : "r"(left_pair), "r"(right_pair));
    return sum_pair;
}
// Fused multiply-add rounds once per FP16 lane; separate multiply and add
// instructions would round twice and are not an equivalent implementation.
__device__ __forceinline__ uint32_t fused_multiply_add_half_pairs(uint32_t left_pair,
                                                                  uint32_t right_pair,
                                                                  uint32_t addend_pair) {
    uint32_t result_pair;
    asm("fma.rn.f16x2 %0,%1,%2,%3;"
        : "=r"(result_pair)
        : "r"(left_pair), "r"(right_pair), "r"(addend_pair));
    return result_pair;
}
__device__ __forceinline__ uint32_t minimum_half_pairs(uint32_t operand_a, uint32_t operand_b) {
#if __CUDA_ARCH__ < 800
    const __half2_raw a{uint16_t(operand_a), uint16_t(operand_a >> 16)};
    const __half2_raw b{uint16_t(operand_b), uint16_t(operand_b >> 16)};
    const auto result = static_cast<__half2_raw>(__hmin2(__half2(a), __half2(b)));
    return uint32_t(result.x) | (uint32_t(result.y) << 16);
#else
    uint32_t destination_words;
    asm("min.f16x2 %0,%1,%2;" : "=r"(destination_words) : "r"(operand_a), "r"(operand_b));
    return destination_words;
#endif
}
__device__ __forceinline__ uint32_t maximum_half_pairs(uint32_t operand_a, uint32_t operand_b) {
#if __CUDA_ARCH__ < 800
    const __half2_raw a{uint16_t(operand_a), uint16_t(operand_a >> 16)};
    const __half2_raw b{uint16_t(operand_b), uint16_t(operand_b >> 16)};
    const auto result = static_cast<__half2_raw>(__hmax2(__half2(a), __half2(b)));
    return uint32_t(result.x) | (uint32_t(result.y) << 16);
#else
    uint32_t destination_words;
    asm("max.f16x2 %0,%1,%2;" : "=r"(destination_words) : "r"(operand_a), "r"(operand_b));
    return destination_words;
#endif
}
// Convert a half pair to two saturating E4M3 bytes in the low 16 bits. The
// caller supplies the upper pair of FP8 bytes to assemble a four-channel word.
__device__ __forceinline__ uint32_t pack_fp8_bits(uint32_t half_pair, uint32_t upper_fp8_pair) {
    uint16_t lower_fp8_pair;
#if __CUDA_ARCH__ < 890
    lower_fp8_pair = phoenix_architecture::encode_fp8_pair(half_pair);
#else
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0,%1;" : "=h"(lower_fp8_pair) : "r"(half_pair));
#endif
    return uint32_t(lower_fp8_pair) | (upper_fp8_pair << 16);
}
__device__ __forceinline__ uint32_t unpack_fp8_bits(uint32_t lower_fp8_pair) {
    uint32_t half_pair;
#if __CUDA_ARCH__ < 890
    half_pair = phoenix_architecture::decode_fp8_pair(uint16_t(lower_fp8_pair));
#else
    asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(half_pair) : "h"(uint16_t(lower_fp8_pair)));
#endif
    return half_pair;
}
// Warp-wide 16x8 matrix product, reducing 32 E4M3 channels into FP16
// accumulators. Fragment words describe each lane's share of the matrices;
// they are not independent scalar rows and cannot be reordered arbitrarily.
__device__ __forceinline__ void
multiply_accumulate_half(uint32_t& result_pair_0, uint32_t& result_pair_1,
                         uint32_t activation_fragment_0, uint32_t activation_fragment_1,
                         uint32_t activation_fragment_2, uint32_t activation_fragment_3,
                         uint32_t weight_fragment_0, uint32_t weight_fragment_1,
                         uint32_t accumulator_input_0, uint32_t accumulator_input_1) {
#if __CUDA_ARCH__ < 890
    phoenix_architecture::fp8_matrix(result_pair_0, result_pair_1,
        activation_fragment_0, activation_fragment_1, activation_fragment_2, activation_fragment_3,
        weight_fragment_0, weight_fragment_1, accumulator_input_0, accumulator_input_1);
#else
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};"
        : "=r"(result_pair_0), "=r"(result_pair_1)
        : "r"(activation_fragment_0), "r"(activation_fragment_1), "r"(activation_fragment_2),
          "r"(activation_fragment_3), "r"(weight_fragment_0), "r"(weight_fragment_1),
          "r"(accumulator_input_0), "r"(accumulator_input_1));
#endif
}
