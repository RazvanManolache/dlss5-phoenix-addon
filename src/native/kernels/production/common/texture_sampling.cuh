#include "architecture_policy.cuh"
// Native frame-boundary sampling and feature packing helpers.
// Memory access and interpolation use the launch texture-coordinate convention.

#define DLSS5_PIPELINE_STAGE -1
#include <cuda_fp16.h>
#include "pipeline_memory.cuh"
__device__ __forceinline__ uint32_t float_minmax(uint32_t operand_a, uint32_t operand_b,
                                                 bool choose_min) {
    float result_word;
    if (choose_min)
        asm("min.ftz.f32 %0,%1,%2;"
            : "=f"(result_word)
            : "f"(__uint_as_float(operand_a)), "f"(__uint_as_float(operand_b)));
    else
        asm("max.ftz.f32 %0,%1,%2;"
            : "=f"(result_word)
            : "f"(__uint_as_float(operand_a)), "f"(__uint_as_float(operand_b)));
    return __float_as_uint(result_word);
}
#define IMAGE_MUFU(name, op)                                                                       \
    __device__ __forceinline__ uint32_t image_mufu_##name(uint32_t operand_bits) {                 \
        float decoded_value;                                                                       \
        asm(op " %0,%1;" : "=f"(decoded_value) : "f"(__uint_as_float(operand_bits)));              \
        return __float_as_uint(decoded_value);                                                     \
    }
IMAGE_MUFU(lg2, "lg2.approx.ftz.f32")
IMAGE_MUFU(ex2, "ex2.approx.ftz.f32")
IMAGE_MUFU(sqrt, "sqrt.approx.ftz.f32")

__device__ __forceinline__ uint32_t image_mufu_cos(uint32_t operand_a) {
    float result_word;
    float radians = __uint_as_float(operand_a) * 6.283185307179586f;
    asm("cos.approx.ftz.f32 %0,%1;" : "=f"(result_word) : "f"(radians));
    return __float_as_uint(result_word);
}
__device__ __forceinline__ uint32_t image_mufu_sin(uint32_t operand_a) {
    float result_word;
    float radians = __uint_as_float(operand_a) * 6.283185307179586f;
    asm("sin.approx.ftz.f32 %0,%1;" : "=f"(result_word) : "f"(radians));
    return __float_as_uint(result_word);
}
__device__ __forceinline__ void image_tex(uint32_t* destination_words, uint64_t handle,
                                          uint32_t address_value, uint32_t word_y) {
#ifdef DLSS5_D3D12_SAMPLER

    asm("tex.2d.v4.f32.f32 {%0,%1,%2,%3},[%4,%5,{%6,%7}];"
        : "=f"(reinterpret_cast<float&>(destination_words[0])),
          "=f"(reinterpret_cast<float&>(destination_words[1])),
          "=f"(reinterpret_cast<float&>(destination_words[2])),
          "=f"(reinterpret_cast<float&>(destination_words[3]))
        : "l"(uint64_t(uint32_t(handle))), "l"(handle >> 32), "f"(__uint_as_float(address_value)),
          "f"(__uint_as_float(word_y)));
#else
    asm("tex.2d.v4.f32.f32 {%0,%1,%2,%3},[%4,{%5,%6}];"
        : "=f"(reinterpret_cast<float&>(destination_words[0])),
          "=f"(reinterpret_cast<float&>(destination_words[1])),
          "=f"(reinterpret_cast<float&>(destination_words[2])),
          "=f"(reinterpret_cast<float&>(destination_words[3]))
        : "l"(handle), "f"(__uint_as_float(address_value)), "f"(__uint_as_float(word_y)));
#endif
}
__device__ __forceinline__ void image_surface(uint32_t handle, uint32_t address_value,
                                              uint32_t word_y, uint32_t operand_a,
                                              uint32_t operand_b, uint32_t operand_c,
                                              uint32_t destination_words) {
    asm volatile("sust.p.2d.v4.b32.zero [%0,{%1,%2}],{%3,%4,%5,%6};" ::"l"(uint64_t(handle)),
                 "r"(address_value), "r"(word_y), "r"(operand_a), "r"(operand_b), "r"(operand_c),
                 "r"(destination_words)
                 : "memory");
}
__device__ __forceinline__ void
image_mma_half(uint32_t& result_pair_0, uint32_t& result_pair_1, uint32_t activation_fragment_0,
               uint32_t activation_fragment_1, uint32_t activation_fragment_2,
               uint32_t activation_fragment_3, uint32_t weight_fragment_0,
               uint32_t weight_fragment_1, uint32_t accumulator_input_0,
               uint32_t accumulator_input_1) {
#if PHOENIX_STYLE_ARCH < 800
    phoenix_architecture::half_matrix(result_pair_0, result_pair_1,
        activation_fragment_0, activation_fragment_1, activation_fragment_2, activation_fragment_3,
        weight_fragment_0, weight_fragment_1, accumulator_input_0, accumulator_input_1);
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};"
        : "=r"(result_pair_0), "=r"(result_pair_1)
        : "r"(activation_fragment_0), "r"(activation_fragment_1), "r"(activation_fragment_2),
          "r"(activation_fragment_3), "r"(weight_fragment_0), "r"(weight_fragment_1),
          "r"(accumulator_input_0), "r"(accumulator_input_1));
#endif
}
template <bool CheckBounds>
__device__ __forceinline__ uint32_t image_read_half(uint64_t address,
                                                    const MemoryRange* memory_ranges,
                                                    uint32_t* error_counts) {
    if constexpr (CheckBounds) {
        bool valid = false;
        for (int range_index = 0; range_index < 16; ++range_index)
            valid |=
                address >= memory_ranges[range_index].base &&
                address - memory_ranges[range_index].base + 2 <= memory_ranges[range_index].size;
        if (!valid) {
            atomicAdd(error_counts, 1u);
            return 0;
        }
    }
    return *reinterpret_cast<const volatile uint16_t*>(address);
}
