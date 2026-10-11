// DLSS5 Phoenix optimized kernel: window_attention/channels_128/attention.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 9216
#include "../../../common/async_memory.cuh"
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y, uint32_t& word_z,
                       uint32_t& word_w, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4* word_pointer = reinterpret_cast<const uint4*>(address);
    uint4 result_word = __ldca(word_pointer);
    word_x = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    word_w = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words_global_cached(uint64_t address, uint32_t& word_x, uint32_t& word_y,
                              uint32_t& word_z, uint32_t& word_w, const MemoryRange* memory_ranges,
                              uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        word_x = word_y = word_z = word_w = 0;
        return;
    }
    const uint4* word_pointer = reinterpret_cast<const uint4*>(address);
    uint4 result_word = __ldcg(word_pointer);
    word_x = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    word_w = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words_streaming(uint64_t address, uint32_t word_x, uint32_t word_y, uint32_t word_z,
                           uint32_t word_w, const MemoryRange* memory_ranges,
                           uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    uint4* word_pointer = reinterpret_cast<uint4*>(address);
    __stcs(word_pointer, make_uint4(word_x, word_y, word_z, word_w));
}

template <bool CheckBounds>
__device__ __forceinline__ uint32_t read_device_flag(uint64_t address,
                                                     const MemoryRange* memory_ranges,
                                                     uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, false)) {
            atomicAdd(error_counts, 1u);
            return 0;
        }
    uint32_t value;
    asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}
template <bool CheckBounds>
__device__ __forceinline__ void write_device_flag(uint64_t address, uint32_t value,
                                                  const MemoryRange* memory_ranges,
                                                  uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, true)) {
            atomicAdd(error_counts + 1, 1u);
            return;
        }
    asm volatile("st.relaxed.gpu.global.u32 [%0],%1;" ::"l"(address), "r"(value) : "memory");
}
// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_output_low_2 = 2,
    parameter_output_high_3 = 3,
    parameter_trained_weights_low_4 = 4,
    parameter_trained_weights_high_5 = 5,
    parameter_tensor_height_8 = 8,
    parameter_tensor_width_9 = 9,
    parameter_geometry_or_control_10 = 10,
    parameter_geometry_or_control_11 = 11,
    parameter_wait_optional_low_12 = 12,
    parameter_wait_optional_high_13 = 13,
    parameter_geometry_or_control_14 = 14,
    parameter_geometry_or_control_15 = 15,
    parameter_publication_optional_low_16 = 16,
    parameter_publication_optional_high_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[22];
};
static_assert(sizeof(KernelArguments) == 88, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_128_attention(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t matrix_fragment_8 = 0;
    uint32_t matrix_fragment_9 = 0;
    uint32_t matrix_fragment_10 = 0;
    uint32_t matrix_fragment_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t matrix_fragment_16 = 0;
    uint32_t matrix_fragment_17 = 0;
    uint32_t matrix_fragment_18 = 0;
    uint32_t matrix_fragment_19 = 0;
    uint32_t matrix_fragment_20 = 0;
    uint32_t matrix_fragment_21 = 0;
    uint32_t matrix_fragment_22 = 0;
    uint32_t matrix_fragment_23 = 0;
    uint32_t matrix_fragment_24 = 0;
    uint32_t matrix_fragment_25 = 0;
    uint32_t matrix_fragment_26 = 0;
    uint32_t matrix_fragment_27 = 0;
    uint32_t matrix_fragment_28 = 0;
    uint32_t matrix_fragment_29 = 0;
    uint32_t matrix_fragment_30 = 0;
    uint32_t matrix_fragment_31 = 0;
    uint32_t matrix_fragment_32 = 0;
    uint32_t matrix_fragment_33 = 0;
    uint32_t matrix_fragment_34 = 0;
    uint32_t matrix_fragment_35 = 0;
    uint32_t matrix_fragment_36 = 0;
    uint32_t matrix_fragment_37 = 0;
    uint32_t matrix_fragment_38 = 0;
    uint32_t matrix_fragment_39 = 0;
    uint32_t matrix_fragment_40 = 0;
    uint32_t matrix_fragment_41 = 0;
    uint32_t matrix_fragment_42 = 0;
    uint32_t matrix_fragment_43 = 0;
    uint32_t matrix_accumulator_44 = 0;
    uint32_t matrix_accumulator_45 = 0;
    uint32_t matrix_accumulator_46 = 0;
    uint32_t matrix_accumulator_47 = 0;
    uint32_t matrix_fragment_48 = 0;
    uint32_t matrix_fragment_49 = 0;
    uint32_t matrix_fragment_50 = 0;
    uint32_t matrix_fragment_51 = 0;
    uint32_t matrix_accumulator_52 = 0;
    uint32_t matrix_accumulator_53 = 0;
    uint32_t matrix_accumulator_54 = 0;
    uint32_t matrix_accumulator_55 = 0;
    uint32_t matrix_accumulator_56 = 0;
    uint32_t matrix_accumulator_57 = 0;
    uint32_t matrix_accumulator_58 = 0;
    uint32_t matrix_accumulator_59 = 0;
    uint32_t matrix_accumulator_60 = 0;
    uint32_t matrix_accumulator_61 = 0;
    uint32_t matrix_accumulator_62 = 0;
    uint32_t matrix_accumulator_63 = 0;
    uint32_t matrix_accumulator_64 = 0;
    uint32_t matrix_accumulator_65 = 0;
    uint32_t matrix_accumulator_66 = 0;
    uint32_t matrix_accumulator_67 = 0;
    uint32_t matrix_accumulator_68 = 0;
    uint32_t matrix_accumulator_69 = 0;
    uint32_t matrix_accumulator_70 = 0;
    uint32_t matrix_accumulator_71 = 0;
    uint32_t matrix_fragment_72 = 0;
    uint32_t matrix_fragment_73 = 0;
    uint32_t packed_half_74 = 0;
    uint32_t matrix_fragment_75 = 0;
    uint32_t matrix_fragment_76 = 0;
    uint32_t matrix_fragment_77 = 0;
    uint32_t matrix_fragment_78 = 0;
    uint32_t matrix_fragment_79 = 0;
    uint32_t matrix_fragment_80 = 0;
    uint32_t matrix_fragment_81 = 0;
    uint32_t matrix_fragment_82 = 0;
    uint32_t matrix_fragment_83 = 0;
    uint32_t matrix_fragment_84 = 0;
    uint32_t matrix_fragment_85 = 0;
    uint32_t matrix_fragment_86 = 0;
    uint32_t matrix_fragment_87 = 0;
    uint32_t matrix_fragment_88 = 0;
    uint32_t matrix_fragment_89 = 0;
    uint32_t matrix_fragment_90 = 0;
    uint32_t matrix_fragment_91 = 0;
    uint32_t matrix_fragment_92 = 0;
    uint32_t matrix_fragment_93 = 0;
    uint32_t matrix_fragment_94 = 0;
    uint32_t matrix_fragment_95 = 0;
    uint32_t matrix_fragment_96 = 0;
    uint32_t matrix_fragment_97 = 0;
    uint32_t matrix_fragment_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t matrix_fragment_100 = 0;
    uint32_t matrix_fragment_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t matrix_fragment_103 = 0;
    uint32_t matrix_accumulator_104 = 0;
    uint32_t matrix_accumulator_105 = 0;
    uint32_t matrix_accumulator_106 = 0;
    uint32_t matrix_accumulator_107 = 0;
    uint32_t matrix_accumulator_108 = 0;
    uint32_t matrix_accumulator_109 = 0;
    uint32_t matrix_accumulator_110 = 0;
    uint32_t matrix_accumulator_111 = 0;
    uint32_t matrix_accumulator_112 = 0;
    uint32_t matrix_accumulator_113 = 0;
    uint32_t matrix_accumulator_114 = 0;
    uint32_t matrix_accumulator_115 = 0;
    uint32_t matrix_accumulator_116 = 0;
    uint32_t matrix_accumulator_117 = 0;
    uint32_t matrix_accumulator_118 = 0;
    uint32_t matrix_accumulator_119 = 0;
    uint32_t matrix_accumulator_120 = 0;
    uint32_t matrix_accumulator_121 = 0;
    uint32_t matrix_accumulator_122 = 0;
    uint32_t matrix_accumulator_123 = 0;
    uint32_t matrix_accumulator_124 = 0;
    uint32_t matrix_accumulator_125 = 0;
    uint32_t matrix_accumulator_126 = 0;
    uint32_t matrix_accumulator_127 = 0;
    uint32_t matrix_accumulator_128 = 0;
    uint32_t matrix_accumulator_129 = 0;
    uint32_t packed_half_130 = 0;
    uint32_t packed_half_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t packed_half_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t index_word_138 = 0;
    uint32_t index_word_139 = 0;
    uint32_t index_word_140 = 0;
    uint32_t tile_word_141 = 0;
    uint32_t address_word_142 = 0;
    uint32_t address_word_143 = 0;
    uint32_t address_word_144 = 0;
    uint32_t address_word_145 = 0;
    uint32_t packed_half_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t address_word_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t packed_half_150 = 0;
    uint32_t packed_half_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t float_bits_158 = 0;
    uint32_t packed_half_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t uniform_parameter_word_4 = 0;
    uint32_t uniform_block_index_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_block_index_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_index_word_15 = 0;
    uint32_t uniform_reused_word_16 = 0;
    uint32_t uniform_reused_word_17 = 0;
    uint32_t uniform_index_word_18 = 0;
    uint32_t uniform_parameter_word_19 = 0;
    uint32_t uniform_index_word_20 = 0;
    uint32_t uniform_index_word_21 = 0;
    uint32_t uniform_block_index_22 = 0;
    uint32_t uniform_parameter_word_23 = 0;
    uint32_t uniform_index_word_24 = 0;
    uint32_t uniform_index_word_25 = 0;
    uint32_t uniform_parameter_word_26 = 0;
    uint32_t uniform_parameter_word_27 = 0;
    uint32_t uniform_parameter_word_28 = 0;
    uint32_t uniform_parameter_word_29 = 0;
    uint32_t uniform_parameter_word_30 = 0;
    uint32_t uniform_parameter_word_31 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    bool uniform_condition_2 = false;
    bool uniform_condition_3 = false;
    bool uniform_condition_4 = false;
    bool uniform_condition_5 = false;
    bool uniform_condition_6 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    uniform_index_word_15 = parameters[parameter_geometry_or_control_15];

    uniform_block_index_22 = blockIdx.y;

    index_word_0 = threadIdx.x;

    uniform_parameter_word_19 = parameters[parameter_tensor_height_8];

    matrix_fragment_75 = threadIdx.y;

    uniform_index_word_18 = parameters[parameter_geometry_or_control_14];

    uniform_block_index_11 = blockIdx.x;

    matrix_accumulator_2 = threadIdx.z;

    uniform_parameter_word_4 = parameters[parameter_geometry_or_control_10];
    uniform_block_index_5 = parameters[parameter_geometry_or_control_11];

    uniform_index_word_14 = parameters[parameter_tensor_width_9];

    uniform_index_word_12 = min(int32_t(0u), int32_t(uniform_index_word_15));

    uniform_index_word_6 = 0u;

    uniform_reused_word_16 = 0u;
    uniform_reused_word_17 = 0u;

    uniform_index_word_12 = 0u - uniform_index_word_12 + uniform_parameter_word_19;

    uniform_index_word_9 = min(int32_t(0u), int32_t(uniform_index_word_18));

    uniform_index_word_13 = min(int32_t(0u), int32_t(uniform_parameter_word_4));

    uniform_index_word_10 = (uniform_block_index_22 << 0x3u) + uniform_block_index_5;

    uniform_condition_3 = uint32_t(uniform_index_word_12) != 0u;

    uniform_block_index_11 = (uniform_block_index_11 << 0x3u) + uniform_parameter_word_4;

    uniform_index_word_9 = 0u - uniform_index_word_9 + uniform_index_word_14;

    uniform_index_word_14 = 0u - uniform_index_word_13 + uniform_index_word_14;

    uniform_index_word_13 = uniform_index_word_10 + (0u - uniform_index_word_15);

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);

    uniform_condition_2 = uint32_t(uniform_index_word_9) != 0u;

    uniform_condition_4 = uint32_t(uniform_index_word_14) != 0u;

    uniform_index_word_20 = uniform_index_word_13 + 0x8u;

    uniform_index_word_7 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x2u >> 32) + uniform_index_word_10;

    condition_0 = (index_word_0 | matrix_accumulator_2 | matrix_fragment_75) != 0;

    uniform_index_word_15 = uniform_index_word_13 + 0x7u;

    index_word_0 = max(int32_t(0u), int32_t(uniform_index_word_13));

    if (uniform_condition_3) {
        uniform_index_word_13 = uniform_index_word_12 + 0xffffffffu;
    }

    uniform_index_word_12 = uint32_t(int32_t(uniform_index_word_7) >> 0x2u);

    uniform_index_word_15 =
        uint32_t(combine_words(uniform_index_word_15, 0u) << 0x1du >> 32) + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_20) > int32_t(0u);

    if (uniform_condition_3) {
        uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_13) >> 0x1fu);
    }

    index_word_0 = uint32_t(uint32_t(index_word_0) >> 0x3u);

    if (uniform_condition_4) {
        uniform_index_word_14 = uniform_index_word_14 + 0xffffffffu;
    }

    uniform_index_word_15 = uniform_condition_0 ? uniform_index_word_15 : 0u;

    if (uniform_condition_3) {
        uniform_index_word_7 =
            uint32_t(combine_words(uniform_index_word_7, 0u) << 0x3u >> 32) + uniform_index_word_13;
    }

    uniform_index_word_18 = uniform_block_index_11 + (0u - uniform_index_word_18);

    if (uniform_condition_2) {
        uniform_index_word_9 = uniform_index_word_9 + 0xffffffffu;
    }

    if (uniform_condition_4) {
        uniform_index_word_13 = uint32_t(int32_t(uniform_index_word_14) >> 0x1fu);
    }

    if (uniform_condition_3) {
        uniform_index_word_6 =
            uint32_t(int64_t(int32_t(uniform_index_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    uniform_index_word_8 = uint32_t(int32_t(uniform_block_index_11) >> 0x1fu);

    uniform_index_word_21 = uniform_index_word_18 + 0x8u;

    uniform_parameter_word_19 = uniform_index_word_18 + 0x7u;

    uniform_index_word_24 = min(int32_t(uniform_index_word_15), int32_t(uniform_index_word_6));

    if (uniform_condition_2) {
        uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_9) >> 0x1fu);
    }

    if (uniform_condition_4) {
        uniform_index_word_14 = uint32_t(combine_words(uniform_index_word_13, 0u) << 0x3u >> 32) +
                                uniform_index_word_14;
    }

    uniform_index_word_8 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x2u >> 32) + uniform_block_index_11;

    uniform_parameter_word_19 =
        uint32_t(combine_words(uniform_parameter_word_19, 0u) << 0x1du >> 32) + 0x1u;

    uniform_condition_1 = int32_t(uniform_index_word_21) > int32_t(0u);

    condition_0 = int32_t(index_word_0) >= int32_t(uniform_index_word_24) || condition_0;

    if (uniform_condition_2) {
        uniform_index_word_7 =
            uint32_t(combine_words(uniform_index_word_7, 0u) << 0x3u >> 32) + uniform_index_word_9;
    }

    if (uniform_condition_4) {
        uniform_index_word_14 = uint32_t(int32_t(uniform_index_word_14) >> 0x3u);
    }

    uniform_block_index_5 = 0u;

    uniform_parameter_word_4 = 0u;

    uniform_index_word_13 = uint32_t(int32_t(uniform_index_word_8) >> 0x2u);

    uniform_parameter_word_19 = uniform_condition_1 ? uniform_parameter_word_19 : 0u;

    if (uniform_condition_2) {
        uniform_block_index_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (uniform_condition_4) {
        uniform_parameter_word_4 =
            uniform_index_word_14 * uniform_block_index_22 + uniform_block_index_22;
    }

    matrix_accumulator_2 = max(int32_t(0u), int32_t(uniform_index_word_18));
    if (!condition_0) {
        matrix_fragment_6 = uniform_block_index_5;

        matrix_fragment_4 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

        matrix_fragment_6 = min(int32_t(matrix_fragment_6), int32_t(uniform_parameter_word_19));
#pragma unroll 1
        do {
            condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_6);
            if (!condition_0) {
                matrix_fragment_8 = index_word_0 * uniform_block_index_5;

                matrix_fragment_5 = matrix_fragment_4;
#pragma unroll 1
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_12];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_7 = matrix_fragment_8 + matrix_fragment_5;

                    split_address_words(
                        matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

                    matrix_fragment_7 = read_device_flag<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_6);

                    condition_0 = int32_t(matrix_fragment_7) > -1;
                    if (!condition_0) {
#pragma unroll 1
                        do {

                            __nanosleep(0x40u);

                            matrix_fragment_7 = read_device_flag<CheckBounds>(
                                combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                memory_ranges, error_counts);

                            condition_0 = int32_t(matrix_fragment_7) > -1;

                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            index_word_0 = index_word_0 + 0x1u;

            condition_0 = uint32_t(index_word_0) != uint32_t(uniform_index_word_24);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    uniform_index_word_20 = parameters[parameter_tensor_height_8];
    uniform_index_word_21 = parameters[parameter_tensor_width_9];

    uniform_condition_6 = false;

    uniform_condition_3 = false;

    uniform_condition_5 = false;

    uniform_condition_1 = true;

    uniform_index_word_6 = uniform_index_word_20 & 0xfffffffcu;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_20) >> 0x1fu);

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_21) >> 0x1fu);

    uniform_condition_2 = uint32_t(uniform_index_word_6) != uint32_t(0x4u);

    uniform_index_word_14 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x2u >> 32) + uniform_index_word_20;

    uniform_index_word_8 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x2u >> 32) + uniform_index_word_21;

    uniform_index_word_6 = 0u;

    uniform_index_word_7 = 0u;

    uniform_index_word_14 = uint32_t(int32_t(uniform_index_word_14) >> 0x2u);

    if (uniform_condition_2) {
        uniform_condition_4 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_14);

        uniform_index_word_9 = uniform_index_word_12 + 0x1u;

        uniform_condition_0 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_14);

        uniform_index_word_15 = uniform_index_word_12 + 0x1u;

        uniform_condition_6 =
            int32_t(uniform_index_word_10) < int32_t(0u - 0x3u) || uniform_condition_4;

        uniform_condition_4 = int32_t(uniform_index_word_9) >= int32_t(uniform_index_word_14);

        uniform_condition_3 =
            int32_t(uniform_index_word_10) < int32_t(0u - 0x3u) || uniform_condition_0;

        uniform_condition_0 = int32_t(uniform_index_word_15) >= int32_t(uniform_index_word_14);

        uniform_condition_1 = !uniform_condition_6;

        uniform_condition_5 =
            int32_t(uniform_index_word_10) < int32_t(0u - 0x7u) || uniform_condition_4;
    }

    uniform_condition_4 = false;

    if (uniform_condition_2) {
        uniform_condition_4 =
            int32_t(uniform_index_word_10) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    uniform_condition_0 = true;

    uniform_index_word_18 = uniform_condition_6 ? 0x40u : 0u;

    if (uniform_condition_2) {
        uniform_condition_0 = !uniform_condition_3;
    }

    uniform_condition_6 = true;

    if (uniform_condition_2) {
        uniform_condition_6 = !uniform_condition_4;
    }

    uniform_index_word_15 = uint32_t(int32_t(uniform_index_word_8) >> 0x2u);

    uniform_index_word_20 = uint32_t(uniform_condition_0);

    uniform_index_word_9 = 0u;

    uniform_index_word_8 = 0u;

    uniform_condition_0 = true;

    uniform_index_word_25 = uniform_condition_6 ? 0x40u : 0u;

    if (uniform_condition_2) {
        uniform_index_word_6 = uniform_index_word_12 * uniform_index_word_15;

        uniform_index_word_7 = uniform_index_word_12 * uniform_index_word_15;

        uniform_index_word_8 =
            uniform_index_word_12 * uniform_index_word_15 + uniform_index_word_15;

        uniform_index_word_9 =
            uniform_index_word_12 * uniform_index_word_15 + uniform_index_word_15;

        uniform_condition_0 = !uniform_condition_5;
    }

    uniform_condition_6 = int32_t(uniform_index_word_18) != int32_t(0u);

    uniform_index_word_18 = uniform_index_word_21 & 0xfffffffcu;

    uniform_condition_2 = int32_t(uniform_index_word_13) >= int32_t(uniform_index_word_15);

    uniform_parameter_word_23 = uint32_t(uniform_condition_5) << 5;

    uniform_condition_5 = int32_t(uniform_index_word_20) != int32_t(0u);

    uniform_index_word_20 = uniform_condition_6 ? uniform_index_word_13 : 0u;

    uniform_condition_2 =
        int32_t(uniform_block_index_11) > int32_t(0u - 0x4u) && !uniform_condition_2;

    uniform_condition_6 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    uniform_index_word_21 = uniform_index_word_13 + 0x1u;

    uniform_condition_1 = uniform_condition_1 & (uniform_condition_6 | uniform_condition_2);

    uniform_condition_6 = int32_t(uniform_index_word_21) >= int32_t(uniform_index_word_15);

    uniform_block_index_22 = uniform_condition_3 ? uniform_index_word_21 : 0u;

    uniform_condition_3 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_3;

    uniform_condition_6 =
        int32_t(uniform_block_index_11) > int32_t(0u - 0x8u) && !uniform_condition_6;

    condition_0 = uniform_condition_1;

    uniform_condition_3 = uniform_condition_5 & (uniform_condition_3 | uniform_condition_6);

    uniform_condition_5 = int32_t(uniform_parameter_word_23) != int32_t(0u);

    if (uniform_condition_1) {
        uniform_parameter_word_26 = parameters[parameter_input_low_0];
        uniform_parameter_word_27 = parameters[parameter_input_high_1];
    }

    uniform_parameter_word_23 = uniform_condition_5 ? uniform_index_word_13 : 0u;

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    condition_1 = uniform_condition_3;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_2);

    uniform_condition_5 = uint32_t(uniform_index_word_18) != uint32_t(0x4u);

    if (uniform_condition_3) {
        uniform_parameter_word_28 = parameters[parameter_input_low_0];
        uniform_parameter_word_29 = parameters[parameter_input_high_1];
    }

    uniform_index_word_20 = !uniform_condition_5 ? uniform_index_word_20 : uniform_index_word_13;

    uniform_block_index_22 = !uniform_condition_5 ? uniform_block_index_22 : uniform_index_word_21;

    uniform_parameter_word_23 =
        !uniform_condition_5 ? uniform_parameter_word_23 : uniform_index_word_13;

    condition_2 = uniform_condition_0;

    if (uniform_condition_1) {
        uniform_index_word_6 = uniform_index_word_20 + uniform_index_word_6;
    }

    if (uniform_condition_3) {
        uniform_index_word_7 = uniform_block_index_22 + uniform_index_word_7;
    }

    if (!uniform_condition_5) {
        uniform_index_word_21 = uniform_condition_4 ? uniform_index_word_21 : 0u;
    }

    uniform_condition_5 = int32_t(uniform_index_word_25) != int32_t(0u);

    uniform_condition_4 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_4;

    if (uniform_condition_0) {
        uniform_index_word_8 = uniform_parameter_word_23 + uniform_index_word_8;
    }

    index_word_0 = uniform_index_word_6;

    matrix_fragment_4 = uniform_index_word_7;

    uniform_condition_4 = uniform_condition_5 & (uniform_condition_4 | uniform_condition_6);

    if (uniform_condition_0) {
        uniform_block_index_22 = parameters[parameter_input_low_0];
        uniform_parameter_word_23 = parameters[parameter_input_high_1];
    }

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_parameter_word_26, uniform_parameter_word_27));

    if (condition_0) {
        matrix_fragment_7 = index_word_0 * 0x4u + matrix_fragment_75;
    }

    matrix_fragment_6 = uniform_index_word_8;

    index_word_0 = threadIdx.x & 31u;

    condition_3 = uniform_condition_4;

    if (condition_1) {
        matrix_fragment_9 = matrix_fragment_4 * 0x4u + matrix_fragment_75;
    }

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        combine_words(uniform_parameter_word_28, uniform_parameter_word_29));

    if (condition_0) {
        matrix_fragment_7 = matrix_fragment_7 * 0x80u;
    }

    if (!uniform_condition_1) {
        uniform_index_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_4) {
        uniform_parameter_word_30 = parameters[parameter_input_low_0];
        uniform_parameter_word_31 = parameters[parameter_input_high_1];
    }

    if (condition_1) {
        matrix_fragment_9 = matrix_fragment_9 * 0x80u;
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_6 * 0x4u + matrix_fragment_75;
    }

    if (uniform_condition_4) {
        uniform_index_word_9 = uniform_index_word_21 + uniform_index_word_9;
    }

    if (condition_0) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (!uniform_condition_3) {
        uniform_index_word_7 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (!uniform_condition_0) {
        uniform_index_word_8 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        combine_words(uniform_block_index_22, uniform_parameter_word_23));

    if (condition_1) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    matrix_fragment_8 = uniform_index_word_9;

    if (!uniform_condition_4) {
        uniform_index_word_9 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (!uniform_condition_1) {
        uniform_index_word_6 = permute_signed(uniform_index_word_6, uniform_index_word_6, 0x5410u);
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 * 0x80u;
    }

    if (!uniform_condition_3) {
        uniform_index_word_7 = permute_signed(uniform_index_word_7, uniform_index_word_7, 0x5410u);
    }

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_8 * 0x4u + matrix_fragment_75;
    }

    if (!uniform_condition_0) {
        uniform_index_word_8 = permute_signed(uniform_index_word_8, uniform_index_word_8, 0x5410u);
    }

    if (!uniform_condition_4) {
        uniform_index_word_9 = permute_signed(uniform_index_word_9, uniform_index_word_9, 0x5410u);
    }

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(uniform_parameter_word_30, uniform_parameter_word_31));

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_13 * 0x80u;
    }

    if (!condition_0) {
        matrix_fragment_12 = uniform_index_word_6;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_0) {
        matrix_fragment_14 = uniform_index_word_6;

        matrix_fragment_15 = uniform_index_word_6;
    }

    if (!condition_1) {
        matrix_fragment_16 = uniform_index_word_7;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_0) {
        matrix_fragment_13 = uniform_index_word_6;
    }

    if (!condition_1) {
        matrix_fragment_17 = uniform_index_word_7;

        matrix_fragment_18 = uniform_index_word_7;

        matrix_fragment_19 = uniform_index_word_7;
    }

    if (!condition_2) {
        matrix_fragment_20 = uniform_index_word_8;

        matrix_fragment_21 = uniform_index_word_8;

        matrix_fragment_22 = uniform_index_word_8;

        matrix_fragment_23 = uniform_index_word_8;
    }

    if (!condition_3) {
        matrix_fragment_24 = uniform_index_word_9;

        matrix_fragment_25 = uniform_index_word_9;

        matrix_fragment_26 = uniform_index_word_9;

        matrix_fragment_27 = uniform_index_word_9;
    }

    if (condition_0) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    __syncthreads();

    if (condition_1) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_3) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);
    }

    uniform_index_word_7 = 0u;

    uniform_index_word_6 = 0x400u;

    uniform_index_word_8 = parameters[parameter_trained_weights_low_4];
    uniform_index_word_9 = parameters[parameter_trained_weights_high_5];

    index_word_140 = matrix_fragment_75 * 0x20u;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    matrix_accumulator_68 = 0;
    matrix_accumulator_69 = 0;

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_fragment_72 = 0;
    matrix_fragment_73 = 0;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_accumulator_66 = 0;
    matrix_accumulator_67 = 0;

    matrix_accumulator_70 = 0;
    matrix_accumulator_71 = 0;

    matrix_accumulator_64 = 0;
    matrix_accumulator_65 = 0;

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    split_address_words(matrix_fragment_84, matrix_fragment_85,
                        combine_words(uniform_index_word_8, uniform_index_word_9));

    uniform_index_word_7 = (uniform_index_word_7 << 0x18u) + uniform_index_word_6;

    uniform_index_word_6 = 0u;

    tile_word_141 = (matrix_fragment_75 << 0x9u) + uniform_index_word_7;

    tile_word_141 = index_word_0 * 0x10u + tile_word_141;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_16,
                                         matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1000u, matrix_fragment_20,
                                         matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    __syncthreads();
#pragma unroll 1
    do {

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            uint64_t(uint32_t(matrix_fragment_75)) * uint64_t(uint32_t(0x4000u)) +
                                combine_words(matrix_fragment_84, matrix_fragment_85));

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_20, matrix_fragment_21),
                                            matrix_fragment_88, matrix_fragment_89,
                                            matrix_fragment_90, matrix_fragment_91, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x200ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x1000ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x1200ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x2000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x2200ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x3000ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x3200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        packed_half_74 = (index_word_0 << 0x4u) + uniform_index_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74, matrix_fragment_28, matrix_fragment_29, matrix_fragment_30,
            matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x800u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1800u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x200u, matrix_fragment_20, matrix_fragment_21,
            matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0xa00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1a00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_80, matrix_fragment_81, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_82, matrix_fragment_83, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1200u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_82, matrix_fragment_83, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_80, matrix_fragment_81, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_80, matrix_fragment_81, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_82, matrix_fragment_83, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x400u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_100,
                                 matrix_fragment_101);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_106,
                                 matrix_accumulator_107);

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            uint64_t(uint32_t(matrix_fragment_75)) * uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_fragment_84, matrix_fragment_85));

        split_address_words(matrix_fragment_100, matrix_fragment_101,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_100, matrix_fragment_101) + 0x10000ull,
            matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_86,
                                 matrix_fragment_87);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_82,
                                 matrix_fragment_83);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_100, matrix_fragment_101) + 0x10200ull,
            matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_110,
                                 matrix_accumulator_111);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x600u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1400u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0xe00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1c00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_86,
                                 matrix_fragment_87);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_94,
                                 matrix_fragment_95);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_92,
                                 matrix_fragment_93);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1600u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        matrix_fragment_92 = 0x2b28u;

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                                 matrix_fragment_89);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_82,
                                 matrix_fragment_83);

        matrix_fragment_96 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_96),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_80,
                                 matrix_fragment_81);

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_104,
                                 matrix_accumulator_105);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96),
            select_half_operand<2, false, false>(matrix_fragment_97), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_98,
                                 matrix_fragment_99);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_74 + 0x1e00u, matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_110,
                                 matrix_accumulator_111);

        matrix_fragment_93 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                select_half_operand<2, false, false>(matrix_fragment_96));

        matrix_fragment_95 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82), 0x44004400u);

        matrix_fragment_95 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_106,
                                 matrix_accumulator_107);

        packed_half_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_95),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_112,
                                 matrix_accumulator_113);

        packed_half_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95),
            select_half_operand<2, false, false>(packed_half_74), 0x3b283b28u);

        matrix_fragment_82 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                select_half_operand<2, false, false>(packed_half_74));

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_8, matrix_fragment_9, matrix_fragment_76,
                                 matrix_fragment_77);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_78,
                                 matrix_fragment_79);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

        matrix_fragment_38 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        packed_half_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_80,
                                 matrix_fragment_81);

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

        matrix_fragment_13 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_96 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_83), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

        matrix_fragment_96 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96), 0xc400c400u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_fragment_97));

        matrix_fragment_97 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_86), 0x44004400u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_13),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_88,
                                 matrix_fragment_89);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_97), 0xc400c400u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

        matrix_fragment_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_96),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_74),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                select_half_operand<2, false, false>(matrix_fragment_36));

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96),
            select_half_operand<2, false, false>(matrix_fragment_97), 0x3b283b28u);

        matrix_fragment_14 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_74),
            select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

        packed_half_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_96 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_74), 0xc400c400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_83 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                select_half_operand<2, false, false>(matrix_fragment_97));

        matrix_fragment_97 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_97), 0xc400c400u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_8, matrix_fragment_9, matrix_fragment_16,
                                 matrix_fragment_17);

        matrix_fragment_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_96),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

        matrix_fragment_38 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

        matrix_fragment_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96),
            select_half_operand<2, false, false>(matrix_fragment_97), 0x3b283b28u);

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_76), 0x44004400u);

        matrix_fragment_100 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_87), 0x44004400u);

        matrix_fragment_9 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_18,
                                 matrix_fragment_19);

        matrix_fragment_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

        matrix_fragment_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                select_half_operand<2, false, false>(matrix_fragment_97));

        matrix_fragment_100 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_100), 0xc400c400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_14),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80), 0x44004400u);

        matrix_fragment_8 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_81), 0x44004400u);

        packed_half_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_77), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14),
            select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78), 0x44004400u);

        matrix_fragment_101 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_100),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_fragment_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        packed_half_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(packed_half_74), 0x3b283b28u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_79), 0x44004400u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_9),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_13 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_101 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_100),
            select_half_operand<2, false, false>(matrix_fragment_101), 0x3b283b28u);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_29 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                select_half_operand<2, false, false>(packed_half_74));

        packed_half_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9),
            select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_87 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                select_half_operand<2, false, false>(matrix_fragment_101));

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                select_half_operand<2, false, false>(matrix_fragment_95));

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_33),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_13),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_15 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_77 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_11 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

        matrix_fragment_37 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                          select_half_operand<2, false, false>(matrix_fragment_97));

        matrix_fragment_28 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_13 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11), 0xc400c400u);

        matrix_fragment_34 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_74),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_28),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_13),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_34),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_74),
            select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

        matrix_fragment_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                select_half_operand<2, false, false>(matrix_fragment_95));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34),
            select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                select_half_operand<2, false, false>(matrix_fragment_36));

        matrix_fragment_92 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                select_half_operand<2, false, false>(matrix_fragment_92));

        matrix_fragment_80 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_29 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                          select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                          select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_81 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                          select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_82 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_83 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                          select_half_operand<2, false, false>(matrix_fragment_92));

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_54,
                                 matrix_accumulator_55);

        uniform_condition_0 = uint32_t(uniform_index_word_6) >= uint32_t(0x60u);

        uniform_block_index_5 = uniform_index_word_6 + 0x20u;

        multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_70,
                                 matrix_accumulator_71);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_72,
                                 matrix_fragment_73);

        multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_68,
                                 matrix_accumulator_69);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_46,
                                 matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_66,
                                 matrix_accumulator_67);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_44,
                                 matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_64,
                                 matrix_accumulator_65);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_2,
                                 matrix_accumulator_3);

        split_address_words(matrix_fragment_84, matrix_fragment_85,
                            combine_words(matrix_fragment_84, matrix_fragment_85) + 0x400u);

        uniform_index_word_6 = uniform_block_index_5;

    } while (!uniform_condition_0);

    address_word_144 = parameters[parameter_trained_weights_low_4];
    address_word_145 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_4 = index_word_140 + 0x8u;

    index_word_138 = index_word_0;

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_6 = index_word_140 + 0x10u;

    matrix_fragment_8 = index_word_140 + 0x18u;

    matrix_fragment_4 = uint32_t(uint32_t(matrix_fragment_4) >> 0x1u);

    index_word_139 = index_word_138 & 0xfffffffcu;

    matrix_fragment_9 = index_word_138 & 0x7ffffffcu;

    matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_6) >> 0x1u);

    matrix_fragment_5 = matrix_fragment_4 + index_word_0 + (0u - index_word_139);

    matrix_fragment_11 = 0u - matrix_fragment_9 + index_word_0;

    matrix_fragment_7 = matrix_fragment_6 + index_word_0 + (0u - index_word_139);

    matrix_fragment_8 = uint32_t(uint32_t(matrix_fragment_8) >> 0x1u);

    matrix_fragment_11 = matrix_fragment_11 * 0x2u + index_word_140;

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_13 = matrix_fragment_8 + index_word_0 + (0u - index_word_139);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_7)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x18010ull, memory_ranges,
        error_counts);

    matrix_fragment_11 = uint32_t(int32_t(matrix_fragment_11) >> 0x1u);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_13)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x18010ull, memory_ranges,
        error_counts);

    split_address_words(packed_half_146, address_word_147,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x18010ull, memory_ranges,
        error_counts);

    matrix_fragment_38 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_146, address_word_147) + 0x18010ull, memory_ranges, error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                                      select_half_operand<2, false, false>(0u));

    uniform_block_index_5 = 0u;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_6 = uniform_index_word_7;

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1800u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    __syncthreads();

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_68 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                      select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_69 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                      select_half_operand<2, false, false>(matrix_fragment_30));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_71 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1000u, matrix_fragment_32,
                                         matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_17);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_141 + 0x1800u, matrix_accumulator_68, matrix_accumulator_69,
        matrix_accumulator_70, matrix_accumulator_71, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_22);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_20);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_21);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_fragment_75)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x14200u);

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    __syncthreads();
#pragma unroll 1
    do {

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5) - 0x200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                            matrix_fragment_36, matrix_fragment_37,
                                            matrix_fragment_38, matrix_fragment_39, memory_ranges,
                                            error_counts);

        matrix_accumulator_46 = (index_word_0 << 0x4u) + uniform_index_word_6;

        uniform_condition_0 = uint32_t(uniform_block_index_5) >= uint32_t(0x60u);

        uniform_index_word_8 = uniform_block_index_5 + 0x20u;

        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x1000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_46, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_6 = uniform_index_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_46 + 0x800u,
                                            matrix_accumulator_52, matrix_accumulator_53,
                                            matrix_accumulator_54, matrix_accumulator_55,
                                            shared_bytes, memory_ranges, error_counts);

        uniform_block_index_5 = uniform_index_word_8;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_46 + 0x1000u,
                                            matrix_accumulator_56, matrix_accumulator_57,
                                            matrix_accumulator_58, matrix_accumulator_59,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_46 + 0x1800u,
                                            matrix_accumulator_60, matrix_accumulator_61,
                                            matrix_accumulator_62, matrix_accumulator_63,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_32, matrix_fragment_33,
                                 matrix_fragment_24, matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_42, matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_36, matrix_fragment_37,
                                 matrix_fragment_16, matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_accumulator_52,
                                 matrix_accumulator_53, matrix_accumulator_54,
                                 matrix_accumulator_55, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_26, matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_32, matrix_fragment_33,
                                 matrix_fragment_6, matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_34, matrix_fragment_35,
                                 matrix_accumulator_2, matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45,
                                 matrix_accumulator_56, matrix_accumulator_57,
                                 matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_36,
                                 matrix_fragment_37, matrix_accumulator_44, matrix_accumulator_45);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_14, matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_32, matrix_fragment_33,
                                 matrix_fragment_18, matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_36, matrix_fragment_37,
                                 matrix_fragment_10, matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_60,
                                 matrix_accumulator_61, matrix_accumulator_62,
                                 matrix_accumulator_63, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_40, matrix_fragment_41);

    } while (!uniform_condition_0);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_64, matrix_accumulator_65,
                        uint64_t(uint32_t(matrix_fragment_75)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(address_word_142, address_word_143,
                        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x18b20u);

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_28,
                                         matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_54 = 0;
    matrix_accumulator_55 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_52 = 0;
    matrix_accumulator_53 = 0;

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_46 = 0;
    matrix_accumulator_47 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                         select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1000u, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_44 = 0;
    matrix_accumulator_45 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1800u, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    matrix_fragment_20 = 0;
    matrix_fragment_21 = 0;

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    packed_half_130 = 0;
    packed_half_131 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_accumulator_132 = 0;
    packed_half_133 = 0;

    uniform_block_index_5 = 0u;

    uniform_index_word_6 = uniform_index_word_7;

    __syncthreads();
#pragma unroll 1
    do {

        split_address_words(address_word_148, packed_half_149,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_142, address_word_143));

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0xa00ull, matrix_accumulator_64,
            matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x800ull, matrix_accumulator_68,
            matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x600ull, matrix_fragment_72,
            matrix_fragment_73, packed_half_74, matrix_fragment_75, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x400ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x200ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(address_word_148, packed_half_149),
                                            matrix_fragment_84, matrix_fragment_85,
                                            matrix_fragment_86, matrix_fragment_87, memory_ranges,
                                            error_counts);

        packed_half_150 = (index_word_0 << 0x4u) + uniform_index_word_6;

        uniform_condition_0 = uint32_t(uniform_block_index_5) >= uint32_t(0x60u);

        uniform_index_word_8 = uniform_block_index_5 + 0x20u;

        split_address_words(address_word_142, address_word_143,
                            combine_words(address_word_142, address_word_143) + 0x3000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_6 = uniform_index_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150 + 0x800u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        uniform_block_index_5 = uniform_index_word_8;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150 + 0x1000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150 + 0x1800u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, packed_half_133, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_64, matrix_accumulator_65,
                                 matrix_accumulator_132, packed_half_133);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_64, matrix_accumulator_65,
                                 matrix_accumulator_104, matrix_accumulator_105);

        multiply_accumulate_half(packed_half_130, packed_half_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_66, matrix_accumulator_67, packed_half_130,
                                 packed_half_131);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_66, matrix_accumulator_67,
                                 matrix_accumulator_106, matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_68, matrix_accumulator_69,
                                 matrix_accumulator_128, matrix_accumulator_129);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_68, matrix_accumulator_69,
                                 matrix_accumulator_44, matrix_accumulator_45);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_68, matrix_accumulator_69,
                                 matrix_accumulator_108, matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_70, matrix_accumulator_71,
                                 matrix_accumulator_134, matrix_accumulator_135);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_70, matrix_accumulator_71,
                                 matrix_accumulator_46, matrix_accumulator_47);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_70, matrix_accumulator_71,
                                 matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_72, matrix_fragment_73,
                                 matrix_accumulator_112, matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 packed_half_74, matrix_fragment_75, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 packed_half_74, matrix_fragment_75, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 packed_half_74, matrix_fragment_75, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, packed_half_74, matrix_fragment_75,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_52,
                                 matrix_accumulator_53);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_76, matrix_fragment_77,
                                 matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_78, matrix_fragment_79,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_80, matrix_fragment_81,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_82, matrix_fragment_83,
                                 matrix_accumulator_122, matrix_accumulator_123);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_86, matrix_fragment_87, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_84, matrix_fragment_85,
                                 matrix_accumulator_124, matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_126, matrix_accumulator_127);

    } while (!uniform_condition_0);

    matrix_fragment_100 = threadIdx.y;

    packed_half_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    condition_0 = (index_word_0 & 0x1u) != 0;

    condition_1 = (index_word_0 & 0x2u) != 0;

    condition_2 = (index_word_0 & 0x8u) != 0;

    condition_3 = (index_word_0 & 0x10u) != 0;

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_11);

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_35);

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_39 = transpose_half_tile(matrix_fragment_39);

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        uint64_t(uint32_t(matrix_fragment_100)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_fragment_38 = transpose_half_tile(matrix_fragment_38);

    matrix_accumulator_59 = transpose_half_tile(matrix_accumulator_59);

    matrix_accumulator_121 = transpose_half_tile(matrix_accumulator_121);

    matrix_accumulator_58 = transpose_half_tile(matrix_accumulator_58);

    matrix_accumulator_125 = transpose_half_tile(matrix_accumulator_125);

    matrix_accumulator_123 = transpose_half_tile(matrix_accumulator_123);

    matrix_accumulator_57 = transpose_half_tile(matrix_accumulator_57);

    matrix_accumulator_61 = transpose_half_tile(matrix_accumulator_61);

    matrix_accumulator_63 = transpose_half_tile(matrix_accumulator_63);

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    matrix_accumulator_122 = transpose_half_tile(matrix_accumulator_122);

    uniform_block_index_5 = blockIdx.y;

    uniform_index_word_6 = parameters[parameter_geometry_or_control_11];

    matrix_fragment_101 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0x2c120ull, memory_ranges,
        error_counts);

    packed_half_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(packed_half_74));

    address_word_143 = matrix_fragment_100 * 0x800u;

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                      select_half_operand<2, false, false>(packed_half_131),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    __syncthreads();

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    split_address_words(address_word_142, address_word_143,
                        uint64_t(uint32_t(address_word_143)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    split_address_words(address_word_142, address_word_143,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_142, address_word_143));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(packed_half_74));

    address_word_148 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24120ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    packed_half_74 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                         select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24320ull, matrix_accumulator_68,
        matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                      select_half_operand<2, false, false>(packed_half_130),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    packed_half_153 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(packed_half_74));

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    packed_half_146 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_146, address_word_147) + 0x30130ull, memory_ranges, error_counts);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_94), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    packed_half_149 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                     select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_102 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    packed_half_150 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_153), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    matrix_fragment_90 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                         select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                     select_half_operand<2, false, false>(packed_half_150));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_102), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_150), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_94), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_103 = result_word;
    }

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                         select_half_operand<2, false, false>(matrix_fragment_83));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_88));

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(matrix_fragment_95));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_151), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_149), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_102), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_153), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_92 = result_word;
    }

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                     select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24520ull, matrix_fragment_72,
        matrix_fragment_73, packed_half_74, matrix_fragment_75, memory_ranges, error_counts);

    packed_half_155 = permute_signed(packed_half_152, packed_half_152, 0x5432u);

    packed_half_154 = permute_signed(matrix_fragment_95, matrix_fragment_95, 0x5432u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24720ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                     select_half_operand<2, false, false>(packed_half_155));

    packed_half_155 = permute_signed(matrix_fragment_94, matrix_fragment_94, 0x5432u);

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(matrix_fragment_82));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_148), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(matrix_fragment_86));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24920ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24b20ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x4100410u);

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                         select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_92 = permute_signed(matrix_fragment_93, matrix_fragment_93, 0x5432u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24d20ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_103 = half_to_float_bits((matrix_fragment_95 & 65535u) * 65537u);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                     select_half_operand<2, false, false>(matrix_fragment_92));

    packed_half_156 = half_to_float_bits((matrix_fragment_95 >> 16) * 65537u);

    packed_half_155 = half_to_float_bits((matrix_fragment_94 & 65535u) * 65537u);

    float_bits_158 = half_to_float_bits((matrix_fragment_94 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x24f20ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    address_word_148 = add_half_pairs(select_half_operand<2, false, false>(address_word_148),
                                      select_half_operand<2, false, false>(matrix_fragment_99));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    packed_half_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_152), 0x4100410u);

    matrix_fragment_103 = rsqrt_bits(matrix_fragment_103);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_159 = result_word;
    }

    packed_half_156 = rsqrt_bits(packed_half_156);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_164 = result_word;
    }

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_99 = half_to_float_bits((packed_half_152 & 65535u) * 65537u);

    packed_half_152 = half_to_float_bits((packed_half_152 >> 16) * 65537u);

    matrix_fragment_99 = rsqrt_bits(matrix_fragment_99);

    packed_half_154 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x4100410u);

    packed_half_152 = rsqrt_bits(packed_half_152);

    packed_half_156 = pack_half(packed_half_156, matrix_fragment_103);

    packed_half_153 = half_to_float_bits((packed_half_154 & 65535u) * 65537u);

    packed_half_154 = half_to_float_bits((packed_half_154 >> 16) * 65537u);

    matrix_fragment_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_131 = permute_signed(matrix_fragment_98, matrix_fragment_98, 0x5432u);

    packed_half_155 = rsqrt_bits(packed_half_155);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(packed_half_159));

    float_bits_158 = rsqrt_bits(float_bits_158);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(packed_half_131));

    packed_half_159 = pack_half(packed_half_152, matrix_fragment_99);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(packed_half_164));

    packed_half_131 = permute_signed(matrix_fragment_102, matrix_fragment_102, 0x5432u);

    packed_half_153 = rsqrt_bits(packed_half_153);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_159));

    matrix_accumulator_128 = permute_signed(matrix_fragment_97, matrix_fragment_97, 0x5432u);

    matrix_fragment_102 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                         select_half_operand<2, false, false>(packed_half_131));

    packed_half_154 = rsqrt_bits(packed_half_154);

    packed_half_131 = permute_signed(matrix_fragment_96, matrix_fragment_96, 0x5432u);

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                       select_half_operand<2, false, false>(matrix_accumulator_128));

    packed_half_155 = pack_half(float_bits_158, packed_half_155);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_151), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    matrix_accumulator_128 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(packed_half_131));

    packed_half_131 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0x4100410u);

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(packed_half_155));

    packed_half_133 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                          select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_4 = half_to_float_bits((packed_half_131 & 65535u) * 65537u);

    packed_half_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                          select_half_operand<2, false, false>(packed_half_159));

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_155));

    packed_half_152 = pack_half(packed_half_154, packed_half_153);

    packed_half_154 = half_to_float_bits((packed_half_131 >> 16) * 65537u);

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_155));

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                          select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_159));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_152));

    packed_half_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_101 = pack_half(0u, matrix_fragment_101);

    packed_half_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    packed_half_133 = rsqrt_bits(packed_half_154);

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(packed_half_159));

    matrix_fragment_102 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(packed_half_155));

    packed_half_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_129 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x4100410u);

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_132 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_149), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_134 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(0u));

    packed_half_131 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                          select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_5 = half_to_float_bits((matrix_accumulator_132 >> 16) * 65537u);

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_136 = half_to_float_bits((matrix_accumulator_132 & 65535u) * 65537u);

    matrix_fragment_98 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                      select_half_operand<2, false, false>(matrix_accumulator_134));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_150), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_157 = result_word;
    }

    packed_half_133 = pack_half(packed_half_133, matrix_fragment_4);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                      select_half_operand<2, false, false>(matrix_accumulator_135));

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                          select_half_operand<2, false, false>(packed_half_152));

    matrix_accumulator_132 = permute_signed(packed_half_151, packed_half_151, 0x5432u);

    packed_half_130 = rsqrt_bits(matrix_accumulator_136);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_4 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x4100410u);

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(matrix_accumulator_132));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_148), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_133 = result_word;
    }

    packed_half_152 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_128 = half_to_float_bits((matrix_fragment_4 & 65535u) * 65537u);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                         select_half_operand<2, false, false>(packed_half_152));

    packed_half_131 = half_to_float_bits((matrix_fragment_4 >> 16) * 65537u);

    packed_half_151 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_151), 0x4100410u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(0u));

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_26 = half_to_float_bits((packed_half_151 & 65535u) * 65537u);

    matrix_accumulator_128 = rsqrt_bits(matrix_accumulator_128);

    packed_half_151 = half_to_float_bits((packed_half_151 >> 16) * 65537u);

    packed_half_130 = pack_half(matrix_fragment_5, packed_half_130);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_131 = rsqrt_bits(packed_half_131);

    matrix_fragment_24 = permute_signed(packed_half_149, packed_half_149, 0x5432u);

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_135 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(packed_half_130));

    matrix_fragment_26 = rsqrt_bits(matrix_fragment_26);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_130));

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_151 = rsqrt_bits(packed_half_151);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_130));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(packed_half_130));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    packed_half_133 = add_half_pairs(select_half_operand<2, false, false>(address_word_148),
                                     select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_29 = permute_signed(packed_half_150, packed_half_150, 0x5432u);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_28 = permute_signed(packed_half_133, packed_half_133, 0x5432u);

    matrix_fragment_27 = pack_half(packed_half_131, matrix_accumulator_128);

    packed_half_150 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_150), 0x4100410u);

    packed_half_133 = add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                     select_half_operand<2, false, false>(matrix_fragment_28));

    packed_half_151 = pack_half(packed_half_151, matrix_fragment_26);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_31 = half_to_float_bits((packed_half_150 & 65535u) * 65537u);

    packed_half_133 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_133), 0x4100410u);

    matrix_accumulator_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    packed_half_149 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_149), 0x4100410u);

    matrix_fragment_50 = rsqrt_bits(matrix_fragment_31);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = half_to_float_bits((matrix_accumulator_129 & 65535u) * 65537u);

    matrix_fragment_7 = half_to_float_bits((matrix_accumulator_129 >> 16) * 65537u);

    matrix_fragment_31 = half_to_float_bits((packed_half_133 & 65535u) * 65537u);

    packed_half_133 = half_to_float_bits((packed_half_133 >> 16) * 65537u);

    packed_half_150 = half_to_float_bits((packed_half_150 >> 16) * 65537u);

    matrix_fragment_30 = half_to_float_bits((packed_half_149 & 65535u) * 65537u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(packed_half_151));

    packed_half_149 = half_to_float_bits((packed_half_149 >> 16) * 65537u);

    matrix_fragment_49 = rsqrt_bits(packed_half_150);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_48 = rsqrt_bits(matrix_fragment_31);

    packed_half_133 = rsqrt_bits(packed_half_133);

    matrix_fragment_6 = rsqrt_bits(matrix_fragment_6);

    matrix_fragment_7 = rsqrt_bits(matrix_fragment_7);

    matrix_fragment_30 = rsqrt_bits(matrix_fragment_30);

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_fragment_49 = pack_half(matrix_fragment_49, matrix_fragment_50);

    matrix_accumulator_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_133 = pack_half(packed_half_133, matrix_fragment_48);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_7 = pack_half(matrix_fragment_7, matrix_fragment_6);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_149 = pack_half(packed_half_149, matrix_fragment_30);

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(packed_half_149));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_53 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_68,
                             matrix_accumulator_69);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(packed_half_74, matrix_fragment_75, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_28, matrix_fragment_29, packed_half_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_96 = 0x29c0u;

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_accumulator_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_accumulator_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    packed_half_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_74),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_68 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3c203c20u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3c203c20u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_accumulator_69 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3c203c20u);

    matrix_accumulator_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_accumulator_68 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_accumulator_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3e473e47u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_accumulator_69 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3e473e47u);

    matrix_accumulator_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3e473e47u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    packed_half_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_74), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    packed_half_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_74), 0x3e473e47u);

    matrix_fragment_21 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_accumulator_68 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_113 = (matrix_accumulator_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_20 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 = (matrix_accumulator_69 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_114 = (matrix_accumulator_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_22 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_23 = (packed_half_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_50 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_51 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                       select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3c203c20u);

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3c203c20u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_81 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3e473e47u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3e473e47u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_75 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    packed_half_74 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_52 = (matrix_fragment_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(packed_half_74));

    matrix_fragment_81 = (matrix_fragment_83 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_53 = (matrix_fragment_85 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_64 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_accumulator_55 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_54 = (matrix_fragment_82 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_fragment_78 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_fragment_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                       select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_66 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_69 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_76 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_fragment_77 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_67 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_68 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_70 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                       select_half_operand<2, false, false>(matrix_accumulator_69));

    matrix_fragment_73 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_71 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                       select_half_operand<2, false, false>(matrix_accumulator_68));

    matrix_fragment_72 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = index_word_0 * 0x4u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_79 = matrix_fragment_18 & 0x1cu;

    matrix_fragment_18 = condition_0 ? matrix_fragment_6 : matrix_fragment_7;

    matrix_fragment_19 = condition_0 ? matrix_fragment_16 : matrix_fragment_17;

    matrix_fragment_6 = condition_0 ? matrix_fragment_7 : matrix_fragment_6;

    matrix_fragment_17 = condition_0 ? matrix_fragment_17 : matrix_fragment_16;

    matrix_fragment_94 =
        uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1du) >> 32) + matrix_fragment_79;

    matrix_fragment_7 = !condition_1 ? matrix_fragment_18 : matrix_fragment_19;

    matrix_fragment_19 = !condition_1 ? matrix_fragment_19 : matrix_fragment_18;

    matrix_fragment_16 = !condition_1 ? matrix_fragment_6 : matrix_fragment_17;

    matrix_fragment_6 = !condition_1 ? matrix_fragment_17 : matrix_fragment_6;

    matrix_fragment_93 = 0x1u ^ matrix_fragment_94;

    matrix_fragment_92 = 0x2u ^ matrix_fragment_94;

    matrix_fragment_17 = 0x3u ^ matrix_fragment_94;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_93), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_19), "r"(matrix_fragment_92), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    matrix_fragment_18 = !condition_2 ? matrix_fragment_7 : matrix_fragment_16;

    matrix_fragment_79 = !condition_2 ? matrix_fragment_16 : matrix_fragment_7;

    matrix_fragment_83 = !condition_2 ? matrix_fragment_19 : matrix_fragment_6;

    matrix_fragment_80 = !condition_2 ? matrix_fragment_6 : matrix_fragment_19;

    matrix_fragment_82 = !condition_3 ? matrix_fragment_18 : matrix_fragment_83;

    matrix_fragment_7 = !condition_3 ? matrix_fragment_79 : matrix_fragment_80;

    matrix_fragment_18 = !condition_3 ? matrix_fragment_83 : matrix_fragment_18;

    matrix_fragment_79 = !condition_3 ? matrix_fragment_80 : matrix_fragment_79;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_7 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_7),
                                       select_half_operand<1, false, false>(matrix_fragment_7));

    matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_7, 0x5410u);

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x4100410u);

    matrix_fragment_19 = uint32_t(int32_t(index_word_138) >> 0x2u);

    matrix_fragment_6 = half_to_float_bits((matrix_fragment_7 & 65535u) * 65537u);

    matrix_fragment_18 = half_to_float_bits((matrix_fragment_7 >> 16) * 65537u);

    matrix_fragment_16 = matrix_fragment_19 + 0x8u;

    matrix_fragment_6 = reciprocal_bits(matrix_fragment_6);

    matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_16) >> 0x1fu);

    matrix_fragment_85 = reciprocal_bits(matrix_fragment_18);

    matrix_fragment_79 =
        uint32_t(combine_words(matrix_fragment_7, 0u) << 0x5u >> 32) + matrix_fragment_16;

    matrix_fragment_80 = uint32_t(int32_t(index_word_138) >> 0x1fu);

    matrix_fragment_79 = matrix_fragment_79 & 0xffffffe0u;

    matrix_fragment_80 =
        uint32_t(combine_words(matrix_fragment_80, 0u) << 0x5u >> 32) + matrix_fragment_19;

    matrix_fragment_88 = matrix_fragment_16 + (0u - matrix_fragment_79);

    matrix_fragment_80 = matrix_fragment_80 & 0xffffffe0u;

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_10);

    matrix_fragment_85 = pack_half(matrix_fragment_85, matrix_fragment_6);

    matrix_fragment_6 = matrix_fragment_19 + (0u - matrix_fragment_80);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_88), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_34);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_89 = uint32_t(int64_t(int32_t(index_word_138)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_90 = uint32_t(int32_t(matrix_fragment_89) >> 0x1fu);

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_90 =
        uint32_t(combine_words(matrix_fragment_90, 0u) << 0x5u >> 32) + matrix_fragment_89;

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_87 = uint32_t(int64_t(int32_t(index_word_138)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_79 = permute_signed(matrix_fragment_88, matrix_fragment_88, 0x5410u);

    matrix_fragment_90 = matrix_fragment_90 & 0xffffffe0u;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_88 = uint32_t(int32_t(matrix_fragment_87) >> 0x1fu);

    matrix_fragment_80 = permute_signed(matrix_fragment_86, matrix_fragment_86, 0x5410u);

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_33);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_90 = matrix_fragment_89 + (0u - matrix_fragment_90);

    matrix_fragment_88 =
        uint32_t(combine_words(matrix_fragment_88, 0u) << 0x5u >> 32) + matrix_fragment_87;

    matrix_fragment_82 = transpose_half_tile(matrix_fragment_8);

    matrix_fragment_83 = transpose_half_tile(matrix_fragment_9);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_84 = transpose_half_tile(matrix_fragment_32);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = transpose_half_tile(matrix_fragment_13);

    matrix_fragment_88 = matrix_fragment_88 & 0xffffffe0u;

    matrix_fragment_86 = transpose_half_tile(matrix_fragment_37);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_20 = transpose_half_tile(matrix_fragment_12);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_90), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_90 = result_word;
    }

    matrix_fragment_88 = matrix_fragment_87 + (0u - matrix_fragment_88);

    matrix_fragment_34 = transpose_half_tile(matrix_fragment_36);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_85), "r"(matrix_fragment_88), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_83 = permute_signed(matrix_fragment_90, matrix_fragment_90, 0x5410u);

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                      select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                      select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_82 = permute_signed(matrix_fragment_88, matrix_fragment_88, 0x5410u);

    matrix_fragment_89 = transpose_half_tile(matrix_accumulator_124);

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_accumulator_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_88 = transpose_half_tile(matrix_accumulator_60);

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_90 = transpose_half_tile(matrix_accumulator_62);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_52),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_51 = transpose_half_tile(matrix_accumulator_120);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_91 = transpose_half_tile(matrix_accumulator_126);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_50 = transpose_half_tile(matrix_accumulator_56);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_53),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_53 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_64 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(0u));

    packed_half_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_74),
                                         select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_accumulator_63));

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_69),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_69 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_accumulator_68 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                      select_half_operand<2, false, false>(matrix_accumulator_69));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_68));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_72));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_fragment_95 = uint32_t(uint32_t(index_word_140) >> 0x1u);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_99 = matrix_fragment_95 + 0x4u;

    matrix_fragment_97 = matrix_fragment_95 + 0x8u;

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_95 = matrix_fragment_95 + 0xcu;

    matrix_fragment_9 = matrix_fragment_99 + index_word_0 + (0u - index_word_139);

    matrix_fragment_11 = matrix_fragment_97 + index_word_0 + (0u - index_word_139);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_accumulator_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_53 = matrix_fragment_95 + index_word_0 + (0u - index_word_139);

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_52, matrix_accumulator_53,
                        uint64_t(uint32_t(matrix_accumulator_53)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_98 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_52, matrix_accumulator_53) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_76,
                                         matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_80,
                                         matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_113 = matrix_fragment_100 * 0x100u;

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        uint64_t(uint32_t(matrix_accumulator_113)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_144, address_word_145));

    split_address_words(matrix_accumulator_112, matrix_accumulator_113,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_112, matrix_accumulator_113));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c130ull,
        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c330ull,
        matrix_accumulator_52, matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d130ull,
        matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d330ull,
        matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e130ull,
        matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e330ull,
        matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f130ull,
        matrix_fragment_72, matrix_fragment_73, packed_half_74, matrix_fragment_75, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f330ull,
        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
        memory_ranges, error_counts);

    matrix_accumulator_122 = (index_word_0 << 0x4u) + uniform_index_word_7;

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_122, matrix_fragment_80,
                                        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_84 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_85 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    uniform_condition_0 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_14);

    uniform_block_index_5 = (uniform_block_index_5 << 0x3u) + uniform_index_word_6;

    uniform_index_word_7 = uniform_index_word_12 * uniform_index_word_15 + uniform_index_word_13;

    uniform_condition_0 =
        int32_t(uniform_block_index_5) > int32_t(0u - 0x4u) && !uniform_condition_0;

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0x800u, matrix_fragment_84, matrix_fragment_85,
        matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_32);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                            select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                            select_half_operand<2, false, false>(packed_half_146));

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0x200u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0xa00u, matrix_fragment_36, matrix_fragment_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_34);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_35);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0xc00u, matrix_fragment_32, matrix_fragment_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_80,
                             matrix_fragment_81);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0x400u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_52,
                             matrix_accumulator_53);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_124,
                             matrix_accumulator_125);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0xe00u, matrix_fragment_36, matrix_fragment_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_accumulator_46));

    matrix_accumulator_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_56));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_accumulator_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_accumulator_57));

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_80,
                             matrix_fragment_81);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_122 + 0x600u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_accumulator_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_accumulator_47));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_accumulator_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_accumulator_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_63));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_124,
                             matrix_accumulator_125);

    matrix_fragment_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_60 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                       select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_accumulator_63 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                       select_half_operand<2, false, false>(matrix_accumulator_63));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_81), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_62 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_60), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_61 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_65 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_63), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_66 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                       select_half_operand<2, false, false>(matrix_accumulator_62));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_74, matrix_fragment_75, matrix_accumulator_118,
                             matrix_accumulator_119);

    matrix_fragment_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                       select_half_operand<2, false, false>(matrix_accumulator_61));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_81), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                       select_half_operand<2, false, false>(matrix_accumulator_66));

    multiply_accumulate_half(packed_half_74, matrix_fragment_75, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             packed_half_74, matrix_fragment_75, matrix_accumulator_124,
                             matrix_accumulator_125);

    uniform_condition_1 = uniform_condition_0 & uniform_condition_2;

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_fragment_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_38 = parameters[parameter_output_low_2];
    matrix_fragment_39 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_accumulator_67 = uniform_index_word_7;

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_52,
                             matrix_accumulator_53);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    matrix_fragment_72 = matrix_fragment_100 * 0x80u;

    matrix_accumulator_61 = matrix_accumulator_67 * 0x200u + matrix_fragment_72;

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_38, matrix_fragment_39,
                        int64_t(int32_t(matrix_accumulator_61)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_38, matrix_fragment_39));
    if (uniform_condition_1) {
        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_53),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_52),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                           select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

        split_address_words(matrix_fragment_32, matrix_fragment_33,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_38, matrix_fragment_39));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_9 = matrix_fragment_34;

        matrix_fragment_11 = matrix_accumulator_55;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_32, matrix_fragment_33), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_6;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_74),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_accumulator_57 = permute_signed(matrix_accumulator_57, matrix_fragment_10, 0x5410u);

        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_38, matrix_fragment_39));

        matrix_accumulator_59 = permute_signed(matrix_accumulator_59, matrix_fragment_8, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_accumulator_56, matrix_fragment_11, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_accumulator_58, matrix_fragment_9, 0x5410u);

        matrix_fragment_9 = matrix_accumulator_59;

        matrix_fragment_11 = matrix_accumulator_57;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) + 0x800ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25120ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25320ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25520ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25720ull, matrix_accumulator_52,
        matrix_accumulator_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25920ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25b20ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25d20ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_142, address_word_143) + 0x25f20ull, matrix_accumulator_68,
        matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    matrix_fragment_81 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_7 =
        uint32_t(combine_words(matrix_fragment_7, 0u) << 0x5u >> 32) + matrix_fragment_16;

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    packed_half_74 = permute_signed(matrix_fragment_81, matrix_fragment_81, 0x5432u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_80 = permute_signed(matrix_fragment_79, matrix_fragment_79, 0x5432u);

    matrix_fragment_81 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(packed_half_74));

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x4100410u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x4100410u);

    packed_half_74 = half_to_float_bits((matrix_fragment_81 & 65535u) * 65537u);

    matrix_fragment_81 = half_to_float_bits((matrix_fragment_81 >> 16) * 65537u);

    matrix_fragment_75 = half_to_float_bits((matrix_fragment_79 & 65535u) * 65537u);

    matrix_fragment_79 = half_to_float_bits((matrix_fragment_79 >> 16) * 65537u);

    packed_half_74 = rsqrt_bits(packed_half_74);

    matrix_fragment_81 = rsqrt_bits(matrix_fragment_81);

    matrix_fragment_75 = rsqrt_bits(matrix_fragment_75);

    matrix_fragment_80 = rsqrt_bits(matrix_fragment_79);

    matrix_fragment_82 = pack_half(matrix_fragment_81, packed_half_74);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_80 = pack_half(matrix_fragment_80, matrix_fragment_75);

    matrix_accumulator_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_40 = permute_signed(matrix_fragment_40, matrix_fragment_75, 0x5410u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_45 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_46 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = permute_signed(matrix_fragment_41, matrix_fragment_42, 0x5410u);

    matrix_fragment_42 = permute_signed(matrix_accumulator_44, matrix_fragment_43, 0x5410u);

    matrix_fragment_43 = permute_signed(matrix_accumulator_45, matrix_accumulator_46, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_9 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_accumulator_46, matrix_accumulator_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x4100410u);

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_76 & 65535u) * 65537u);

    matrix_fragment_10 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_fragment_76 = half_to_float_bits((matrix_fragment_76 >> 16) * 65537u);

    matrix_accumulator_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_accumulator_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_44 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    matrix_fragment_9 = rsqrt_bits(matrix_fragment_76);

    matrix_accumulator_45 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x3c203c20u);

    matrix_accumulator_44 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x3e473e47u);

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_73 & 65535u) * 65537u);

    matrix_accumulator_45 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x3e473e47u);

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_73 >> 16) * 65537u);

    matrix_accumulator_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_46),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_47),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_accumulator_46 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_46), 0x3c203c20u);

    matrix_accumulator_47 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_47), 0x3c203c20u);

    matrix_accumulator_46 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_46), 0x3e473e47u);

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

    matrix_fragment_8 = pack_half(matrix_fragment_9, matrix_fragment_8);

    matrix_accumulator_47 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_47), 0x3e473e47u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3c203c20u);

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3e473e47u);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_9 = pack_half(matrix_fragment_73, matrix_fragment_10);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_33),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_52, matrix_accumulator_53, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_52,
                             matrix_accumulator_53);

    matrix_fragment_9 = permute_signed(matrix_accumulator_105, matrix_fragment_10, 0x5410u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_35),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_10 = permute_signed(matrix_accumulator_108, matrix_fragment_11, 0x5410u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3c203c20u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3c203c20u);

    matrix_fragment_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3c203c20u);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3e473e47u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3e473e47u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3c203c20u);

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3e473e47u);

    matrix_accumulator_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_11 = permute_signed(matrix_accumulator_109, matrix_fragment_40, 0x5410u);

    matrix_accumulator_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_56,
                             matrix_accumulator_57);

    matrix_accumulator_52 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3c203c20u);

    matrix_accumulator_53 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3c203c20u);

    matrix_accumulator_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_52), 0x3e473e47u);

    matrix_accumulator_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_53), 0x3e473e47u);

    matrix_accumulator_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_54 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3c203c20u);

    matrix_accumulator_55 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3e473e47u);

    matrix_accumulator_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3e473e47u);

    matrix_accumulator_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_56 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3c203c20u);

    matrix_accumulator_57 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3c203c20u);

    matrix_accumulator_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3e473e47u);

    matrix_accumulator_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3e473e47u);

    matrix_fragment_4 = (matrix_accumulator_44 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = (matrix_accumulator_45 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_2 = (matrix_accumulator_56 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_accumulator_58 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3c203c20u);

    matrix_accumulator_59 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3c203c20u);

    matrix_accumulator_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3e473e47u);

    matrix_accumulator_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3e473e47u);

    matrix_accumulator_3 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_fragment_27 = (matrix_accumulator_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = (matrix_accumulator_47 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_24 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_46 = (matrix_accumulator_54 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_accumulator_62 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3c203c20u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_accumulator_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3e473e47u);

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    matrix_fragment_28 = (matrix_fragment_32 << 0x5u) + 0x7ff88000u;

    matrix_fragment_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_29 = (matrix_fragment_33 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_25 = (matrix_accumulator_62 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_47 = (matrix_accumulator_55 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_68,
                             matrix_accumulator_69);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_fragment_31 = (matrix_fragment_34 << 0x5u) + 0x7ff88000u;

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                        select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_30 = (matrix_fragment_35 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_fragment_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                        select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_8 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_9 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_11 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_10 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_34 = (matrix_accumulator_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_35 = (matrix_accumulator_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_36 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_37 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_44 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_32 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_45 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_33 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_68 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3c203c20u);

    matrix_accumulator_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_69 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3c203c20u);

    matrix_accumulator_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3c203c20u);

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_accumulator_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3c203c20u);

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                       select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_fragment_38 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_57 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_39 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_68 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3e473e47u);

    matrix_accumulator_69 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3e473e47u);

    matrix_accumulator_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3e473e47u);

    matrix_accumulator_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3e473e47u);

    matrix_accumulator_45 = (matrix_accumulator_52 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_44 = (matrix_accumulator_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 = (matrix_accumulator_68 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_41 = (matrix_accumulator_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                       select_half_operand<2, false, false>(matrix_accumulator_46));

    matrix_fragment_43 = (matrix_accumulator_70 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                       select_half_operand<2, false, false>(matrix_accumulator_47));

    matrix_fragment_42 = (matrix_accumulator_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_55 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                       select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_56),
                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_53 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_57),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_accumulator_52 = condition_0 ? matrix_fragment_49 : matrix_fragment_48;

    matrix_fragment_48 = condition_0 ? matrix_fragment_48 : matrix_fragment_49;

    matrix_fragment_49 = condition_0 ? matrix_accumulator_54 : matrix_accumulator_53;

    matrix_accumulator_53 = condition_0 ? matrix_accumulator_53 : matrix_accumulator_54;

    matrix_accumulator_55 = !condition_1 ? matrix_accumulator_52 : matrix_fragment_49;

    matrix_accumulator_54 = !condition_1 ? matrix_fragment_48 : matrix_accumulator_53;

    matrix_fragment_49 = !condition_1 ? matrix_fragment_49 : matrix_accumulator_52;

    matrix_fragment_48 = !condition_1 ? matrix_accumulator_53 : matrix_fragment_48;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_55), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_55 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_54), "r"(matrix_fragment_93), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_54 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_92), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_48), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_accumulator_52 = !condition_2 ? matrix_accumulator_55 : matrix_accumulator_54;

    matrix_accumulator_53 = !condition_2 ? matrix_accumulator_54 : matrix_accumulator_55;

    matrix_accumulator_57 = !condition_2 ? matrix_fragment_49 : matrix_fragment_48;

    matrix_accumulator_56 = !condition_2 ? matrix_fragment_48 : matrix_fragment_49;

    matrix_fragment_49 = matrix_fragment_7 & 0xffffffe0u;

    matrix_fragment_48 = uint32_t(int64_t(int32_t(index_word_138)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_accumulator_58 = !condition_3 ? matrix_accumulator_52 : matrix_accumulator_57;

    matrix_accumulator_59 = !condition_3 ? matrix_accumulator_53 : matrix_accumulator_56;

    matrix_accumulator_52 = !condition_3 ? matrix_accumulator_57 : matrix_accumulator_52;

    matrix_accumulator_53 = !condition_3 ? matrix_accumulator_56 : matrix_accumulator_53;

    matrix_fragment_16 = matrix_fragment_16 + (0u - matrix_fragment_49);

    matrix_fragment_49 = uint32_t(int64_t(int32_t(index_word_138)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_accumulator_58 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_accumulator_58 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                       select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_52 = uint32_t(int32_t(matrix_fragment_49) >> 0x1fu);

    matrix_accumulator_58 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                       select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_52 =
        uint32_t(combine_words(matrix_accumulator_52, 0u) << 0x5u >> 32) + matrix_fragment_49;

    matrix_accumulator_58 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_58),
                       select_half_operand<1, false, false>(matrix_accumulator_58));

    matrix_accumulator_54 = matrix_accumulator_52 & 0xffffffe0u;

    matrix_accumulator_58 = permute_signed(matrix_accumulator_58, matrix_accumulator_58, 0x5410u);

    matrix_accumulator_54 = matrix_fragment_49 + (0u - matrix_accumulator_54);

    matrix_accumulator_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58), 0x4100410u);

    matrix_fragment_17 = half_to_float_bits((matrix_accumulator_58 & 65535u) * 65537u);

    matrix_accumulator_58 = half_to_float_bits((matrix_accumulator_58 >> 16) * 65537u);

    matrix_fragment_17 = reciprocal_bits(matrix_fragment_17);

    matrix_accumulator_58 = reciprocal_bits(matrix_accumulator_58);

    matrix_fragment_7 = pack_half(matrix_accumulator_58, matrix_fragment_17);

    matrix_fragment_17 = uint32_t(int32_t(matrix_fragment_48) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_16), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    matrix_fragment_17 =
        uint32_t(combine_words(matrix_fragment_17, 0u) << 0x5u >> 32) + matrix_fragment_48;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    matrix_fragment_17 = matrix_fragment_17 & 0xffffffe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_accumulator_54), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_53 = result_word;
    }

    matrix_accumulator_52 = matrix_fragment_48 + (0u - matrix_fragment_17);

    matrix_fragment_49 = permute_signed(matrix_fragment_16, matrix_fragment_16, 0x5410u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_accumulator_52), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    matrix_fragment_48 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5410u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_53 = permute_signed(matrix_accumulator_53, matrix_accumulator_53, 0x5410u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_52 = permute_signed(matrix_fragment_16, matrix_fragment_16, 0x5410u);

    matrix_accumulator_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_47),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_accumulator_3 = index_word_138 & 0x7ffffffcu;

    index_word_138 = index_word_138 & 0xfffffffcu;

    matrix_fragment_99 = matrix_fragment_99 + index_word_0 + (0u - index_word_138);

    matrix_fragment_97 = matrix_fragment_97 + index_word_0 + (0u - index_word_138);

    matrix_fragment_95 = matrix_fragment_95 + index_word_0 + (0u - index_word_138);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_46),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_141 + 0x1000u, matrix_accumulator_44, matrix_accumulator_45,
        matrix_accumulator_46, matrix_accumulator_47, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_5 = 0u - matrix_accumulator_3 + index_word_0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(0u));

    index_word_140 = matrix_fragment_5 * 0x2u + index_word_140;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = uint32_t(int32_t(index_word_140) >> 0x1u);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_20,
                             matrix_fragment_21);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x1800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_97)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_fragment_48 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_53 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_99)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_fragment_95)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_141 + 0x800u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_52 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_54 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x30130ull, memory_ranges,
        error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c130ull,
        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2c330ull,
        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d130ull,
        matrix_fragment_28, matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2d330ull,
        matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e130ull,
        matrix_fragment_16, matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2e330ull,
        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f130ull,
        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_112, matrix_accumulator_113) + 0x2f330ull,
        matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
        error_counts);

    uniform_index_word_7 = 0u;

    uniform_index_word_6 = 0x400u;

    uniform_index_word_6 = (uniform_index_word_7 << 0x18u) + uniform_index_word_6;

    matrix_accumulator_55 = (index_word_0 << 0x4u) + uniform_index_word_6;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55, matrix_accumulator_56, matrix_accumulator_57,
        matrix_accumulator_58, matrix_accumulator_59, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_44);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_45);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x800u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_44 = unpack_fp8_bits(matrix_accumulator_44 >> 16);

    matrix_accumulator_45 = unpack_fp8_bits(matrix_accumulator_45 >> 16);

    uniform_index_word_15 = uniform_index_word_12 * uniform_index_word_15 + uniform_index_word_15;

    uniform_index_word_12 = uniform_index_word_12 + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_14);

    uniform_index_word_13 = uniform_index_word_13 + uniform_index_word_15;

    uniform_condition_0 =
        int32_t(uniform_block_index_5) > int32_t(0u - 0x8u) && !uniform_condition_0;

    uniform_condition_2 = uniform_condition_0 & uniform_condition_2;

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_46);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_47);

    matrix_accumulator_46 = unpack_fp8_bits(matrix_accumulator_46 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_47 = unpack_fp8_bits(matrix_accumulator_47 >> 16);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_46),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_accumulator_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_47),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_32);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_44,
                             matrix_accumulator_45);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x200u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_52));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_46,
                             matrix_accumulator_47);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0xa00u, matrix_accumulator_44, matrix_accumulator_45,
        matrix_accumulator_46, matrix_accumulator_47, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_34);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_53));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_54));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x400u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_accumulator_44,
                             matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_accumulator_44,
                             matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_44,
                             matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_accumulator_44,
                             matrix_accumulator_45, matrix_accumulator_46, matrix_accumulator_47,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_64,
                             matrix_accumulator_65);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0xe00u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_55 + 0x600u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_9 = threadIdx.x;

    matrix_fragment_8 = threadIdx.z;

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_56,
                             matrix_accumulator_57);

    condition_0 = (matrix_fragment_100 | matrix_fragment_8 | matrix_fragment_9) != 0;

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_5 = uniform_index_word_13;

    matrix_fragment_5 = matrix_fragment_5 * 0x200u + matrix_fragment_72;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));
    if (uniform_condition_2) {
        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_6;

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_13 = permute_signed(matrix_fragment_13, matrix_fragment_6, 0x5410u);

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_37 = permute_signed(matrix_fragment_37, matrix_fragment_4, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_12, matrix_fragment_7, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_36, matrix_fragment_5, 0x5410u);

        matrix_fragment_5 = matrix_fragment_37;

        matrix_fragment_7 = matrix_fragment_13;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x800ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_block_index_5 = blockIdx.x;

    uniform_index_word_6 = parameters[parameter_publication_optional_low_16];
    uniform_index_word_7 = parameters[parameter_publication_optional_high_17];

    uniform_block_index_5 = uniform_block_index_5 + uniform_parameter_word_4;

    split_address_words(uniform_index_word_6, uniform_index_word_7,
                        int64_t(int32_t(uniform_block_index_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_index_word_6, uniform_index_word_7));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_index_word_6, uniform_index_word_7));

    __threadfence();

    write_device_flag<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                   memory_ranges, error_counts);
    return;
}
extern "C" __global__ void
phoenix_window_attention_channels_128_attention(KernelArguments parameters) {
    run_window_attention_channels_128_attention<false>(parameters.words, nullptr, nullptr);
}
