// DLSS5 Phoenix optimized kernel: window_attention/channels_128/output_view.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (__CUDA_ARCH__ >= 900)
#define DLSS5_TILE_SHARED_BYTES 9216
#include "../../../common/device_memory_scalar.cuh"
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
    parameter_optional_control_10 = 10,
    parameter_scalar_control_11 = 11,
    parameter_wait_optional_low_12 = 12,
    parameter_wait_optional_high_13 = 13,
    parameter_scalar_control_14 = 14,
    parameter_optional_control_15 = 15,
    parameter_optional_control_20 = 20,
    parameter_optional_control_21 = 21,
};

struct alignas(8) KernelArguments {
    uint32_t words[22];
};
static_assert(sizeof(KernelArguments) == 88, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_128_output_view(
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
    uint32_t matrix_accumulator_36 = 0;
    uint32_t matrix_accumulator_37 = 0;
    uint32_t matrix_fragment_38 = 0;
    uint32_t matrix_fragment_39 = 0;
    uint32_t matrix_fragment_40 = 0;
    uint32_t matrix_fragment_41 = 0;
    uint32_t matrix_fragment_42 = 0;
    uint32_t matrix_fragment_43 = 0;
    uint32_t matrix_fragment_44 = 0;
    uint32_t matrix_fragment_45 = 0;
    uint32_t matrix_fragment_46 = 0;
    uint32_t matrix_fragment_47 = 0;
    uint32_t matrix_accumulator_48 = 0;
    uint32_t matrix_accumulator_49 = 0;
    uint32_t matrix_fragment_50 = 0;
    uint32_t matrix_fragment_51 = 0;
    uint32_t matrix_fragment_52 = 0;
    uint32_t matrix_fragment_53 = 0;
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
    uint32_t matrix_fragment_68 = 0;
    uint32_t matrix_fragment_69 = 0;
    uint32_t matrix_fragment_70 = 0;
    uint32_t matrix_fragment_71 = 0;
    uint32_t matrix_fragment_72 = 0;
    uint32_t matrix_fragment_73 = 0;
    uint32_t matrix_fragment_74 = 0;
    uint32_t matrix_fragment_75 = 0;
    uint32_t matrix_fragment_76 = 0;
    uint32_t matrix_fragment_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
    uint32_t matrix_fragment_80 = 0;
    uint32_t matrix_fragment_81 = 0;
    uint32_t matrix_accumulator_82 = 0;
    uint32_t matrix_accumulator_83 = 0;
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
    uint32_t index_word_128 = 0;
    uint32_t index_word_129 = 0;
    uint32_t matrix_accumulator_130 = 0;
    uint32_t matrix_accumulator_131 = 0;
    uint32_t packed_half_132 = 0;
    uint32_t packed_half_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t matrix_accumulator_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t address_word_140 = 0;
    uint32_t index_word_141 = 0;
    uint32_t index_word_142 = 0;
    uint32_t tile_word_143 = 0;
    uint32_t address_word_144 = 0;
    uint32_t address_word_145 = 0;
    uint32_t address_word_146 = 0;
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
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_parameter_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_parameter_word_9 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool condition_6 = false;
    bool uniform_condition_0 = false;
    bool uniform_condition_1 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    index_word_0 = blockIdx.y;

    uniform_parameter_word_9 = parameters[parameter_optional_control_15];

    matrix_fragment_8 = parameters[parameter_optional_control_10];
    matrix_fragment_9 = parameters[parameter_scalar_control_11];

    matrix_accumulator_3 = blockIdx.x;

    uniform_parameter_word_7 = parameters[parameter_tensor_height_8];

    matrix_fragment_6 = threadIdx.x;

    uniform_index_word_8 = parameters[parameter_scalar_control_14];

    matrix_fragment_89 = threadIdx.y;

    uniform_parameter_word_6 = parameters[parameter_tensor_width_9];

    matrix_fragment_5 = threadIdx.z;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_9));

    uniform_parameter_word_7 = 0u - uniform_index_word_4 + uniform_parameter_word_7;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_8));

    uniform_index_word_4 = 0u;

    uniform_condition_1 = uint32_t(uniform_parameter_word_7) != 0u;

    uniform_parameter_word_6 = 0u - uniform_index_word_5 + uniform_parameter_word_6;

    index_word_0 = index_word_0 * 0x8u + matrix_fragment_9;

    uniform_index_word_5 = 0u;

    uniform_condition_0 = uint32_t(uniform_parameter_word_6) != 0u;

    matrix_fragment_4 = matrix_accumulator_3 * 0x8u + matrix_fragment_8;

    matrix_accumulator_2 = index_word_0 + (0u - uniform_parameter_word_9);

    if (uniform_condition_1) {
        uniform_parameter_word_7 = uniform_parameter_word_7 + 0xffffffffu;
    }

    matrix_fragment_7 = matrix_fragment_4 + (0u - uniform_index_word_8);

    matrix_fragment_8 = matrix_accumulator_2 + 0x8u;

    matrix_accumulator_3 = matrix_accumulator_2 + 0x7u;

    if (uniform_condition_1) {
        uniform_index_word_8 = uint32_t(int32_t(uniform_parameter_word_7) >> 0x1fu);
    }

    matrix_accumulator_2 = max(int32_t(0u), int32_t(matrix_accumulator_2));

    if (uniform_condition_0) {
        uniform_parameter_word_6 = uniform_parameter_word_6 + 0xffffffffu;
    }

    condition_1 = int32_t(matrix_fragment_8) > int32_t(0u);

    condition_0 = (matrix_fragment_5 | matrix_fragment_6 | matrix_fragment_89) != 0;

    if (uniform_condition_1) {
        uniform_index_word_8 = uint32_t(combine_words(uniform_index_word_8, 0u) << 0x3u >> 32) +
                               uniform_parameter_word_7;
    }

    matrix_fragment_5 = uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_parameter_word_7 = uint32_t(int32_t(uniform_parameter_word_6) >> 0x1fu);
    }

    matrix_accumulator_2 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

    if (uniform_condition_1) {
        uniform_index_word_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_8)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_fragment_9 = matrix_fragment_7 + 0x8u;

    matrix_fragment_6 = condition_1 ? matrix_fragment_5 : 0u;

    matrix_fragment_8 = matrix_fragment_7 + 0x7u;

    matrix_accumulator_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_parameter_word_7 =
            uint32_t(combine_words(uniform_parameter_word_7, 0u) << 0x3u >> 32) +
            uniform_parameter_word_6;
    }

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    matrix_fragment_12 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_5));

    condition_2 = int32_t(matrix_fragment_9) > int32_t(0u);

    matrix_fragment_8 = uint32_t(combine_words(matrix_fragment_8, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + index_word_0;

    condition_0 = int32_t(matrix_accumulator_2) >= int32_t(matrix_fragment_12) || condition_0;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_4;

    matrix_fragment_7 = max(int32_t(0u), int32_t(matrix_fragment_7));

    matrix_fragment_10 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    matrix_fragment_11 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    matrix_fragment_8 = condition_2 ? matrix_fragment_8 : 0u;
    if (!condition_0) {
        matrix_fragment_7 = uint32_t(uint32_t(matrix_fragment_7) >> 0x3u);

        matrix_fragment_8 = min(int32_t(matrix_fragment_8), int32_t(uniform_index_word_4));

        matrix_fragment_5 = matrix_accumulator_2;
        do {
            condition_0 = int32_t(matrix_fragment_7) >= int32_t(matrix_fragment_8);
            if (!condition_0) {
                matrix_fragment_13 = matrix_fragment_5 * uniform_index_word_4;

                matrix_fragment_6 = matrix_fragment_7;
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_12];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_9 = matrix_fragment_13 + matrix_fragment_6;

                    split_address_words(
                        matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

                    matrix_fragment_9 = pipeline_read<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    matrix_fragment_6 = matrix_fragment_6 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_6) != uint32_t(matrix_fragment_8);

                    condition_0 = int32_t(matrix_fragment_9) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_9 = pipeline_read<CheckBounds>(
                                combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                memory_ranges, error_counts);

                            condition_0 = int32_t(matrix_fragment_9) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            matrix_fragment_5 = matrix_fragment_5 + 0x1u;

            condition_0 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_12);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    index_word_128 = parameters[parameter_tensor_height_8];
    index_word_129 = parameters[parameter_tensor_width_9];

    condition_4 = false;

    condition_5 = false;

    matrix_fragment_12 = matrix_fragment_11 + 0x1u;

    matrix_fragment_16 = 0u;

    matrix_accumulator_2 = index_word_128 & 0xfffffffcu;

    matrix_accumulator_3 = uint32_t(int32_t(index_word_128) >> 0x1fu);

    matrix_fragment_14 = index_word_129 & 0xfffffffcu;

    condition_0 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + index_word_128;

    matrix_accumulator_2 = uint32_t(int32_t(index_word_129) >> 0x1fu);

    matrix_accumulator_3 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x2u >> 32) + index_word_129;

    matrix_fragment_13 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    if (condition_0) {
        matrix_fragment_6 = matrix_fragment_10 + 0x1u;

        condition_1 = int32_t(matrix_fragment_10) >= int32_t(matrix_accumulator_3);
    }

    condition_3 = int32_t(matrix_fragment_11) >= int32_t(matrix_fragment_13);

    if (condition_0) {
        matrix_fragment_16 = matrix_fragment_10 * matrix_fragment_13 + matrix_fragment_13;

        condition_2 = int32_t(matrix_fragment_6) >= int32_t(matrix_accumulator_3);

        condition_5 = int32_t(index_word_0) < int32_t(0u - 0x3u) || condition_1;
    }

    condition_1 = true;

    condition_3 = int32_t(matrix_fragment_4) > int32_t(0u - 0x4u) && !condition_3;

    if (condition_0) {
        condition_4 = int32_t(index_word_0) < int32_t(0u - 0x7u) || condition_2;

        condition_1 = !condition_5;
    }

    matrix_accumulator_2 = uint32_t(condition_5) << 5;

    matrix_fragment_5 = uint32_t(condition_4) << 4;

    condition_4 = uint32_t(matrix_fragment_14) == uint32_t(0x4u) || condition_5;

    condition_2 = true;

    condition_1 = condition_1 & (condition_4 | condition_3);

    matrix_fragment_8 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_fragment_5) != int32_t(0u);

    condition_4 = uint32_t(matrix_fragment_14) == uint32_t(0x4u) || condition_1;

    if (condition_0) {
        condition_2 = !condition_1;
    }

    matrix_fragment_7 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_accumulator_2) != int32_t(0u);

    if (condition_0) {
        matrix_accumulator_2 = matrix_fragment_10 + 0x1u;
    }

    condition_2 = condition_2 & (condition_4 | condition_3);

    if (condition_0) {
        condition_4 = int32_t(matrix_fragment_10) >= int32_t(matrix_accumulator_3);
    }

    condition_3 = false;

    if (condition_0) {
        condition_5 = int32_t(matrix_accumulator_2) >= int32_t(matrix_accumulator_3);
    }

    matrix_fragment_5 = uint32_t(condition_2) << 2;

    if (condition_0) {
        condition_3 = int32_t(index_word_0) < int32_t(0u - 0x3u) || condition_4;
    }

    condition_4 = false;

    if (condition_0) {
        condition_4 = int32_t(index_word_0) < int32_t(0u - 0x7u) || condition_5;
    }

    condition_5 = int32_t(matrix_fragment_12) >= int32_t(matrix_fragment_13);

    condition_6 = uint32_t(matrix_fragment_14) == uint32_t(0x4u) || condition_3;

    matrix_fragment_6 = uint32_t(condition_1) << 1;

    condition_1 = uint32_t(matrix_fragment_14) == uint32_t(0x4u) || condition_4;

    condition_2 = int32_t(matrix_fragment_4) > int32_t(0u - 0x8u) && !condition_5;

    condition_5 = true;

    if (condition_0) {
        condition_5 = !condition_3;
    }

    matrix_fragment_9 = condition_3 ? matrix_fragment_12 : 0u;

    condition_3 = uint32_t(matrix_fragment_14) != uint32_t(0x4u);

    condition_5 = condition_5 & (condition_6 | condition_2);

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    condition_6 = true;

    if (condition_0) {
        matrix_fragment_14 = matrix_fragment_10 * matrix_fragment_13;

        condition_6 = !condition_4;

        matrix_fragment_15 = matrix_fragment_10 * matrix_fragment_13 + matrix_fragment_13;
    }

    matrix_fragment_19 = !condition_3 ? matrix_fragment_9 : matrix_fragment_12;

    condition_6 = condition_6 & (condition_1 | condition_2);

    condition_1 = int32_t(matrix_fragment_6) != int32_t(0u);

    condition_2 = int32_t(matrix_fragment_5) != int32_t(0u);

    if (!condition_3) {
        matrix_fragment_12 = condition_4 ? matrix_fragment_12 : 0u;
    }

    if (condition_5) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];

        matrix_fragment_14 = matrix_fragment_19 + matrix_fragment_14;
    }

    index_word_0 = condition_1 ? matrix_fragment_11 : 0u;

    condition_1 = int32_t(matrix_fragment_7) != int32_t(0u);

    if (!condition_5) {
        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_17 = !condition_3 ? index_word_0 : matrix_fragment_11;

    if (!condition_3) {
        matrix_fragment_11 = condition_1 ? matrix_fragment_11 : 0u;
    }

    condition_1 = int32_t(matrix_fragment_8) != int32_t(0u);

    index_word_0 = 0u;

    if (condition_0) {
        index_word_0 = matrix_fragment_10 * matrix_fragment_13;
    }

    if (condition_2) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];

        matrix_fragment_16 = matrix_fragment_11 + matrix_fragment_16;
    }

    if (condition_6) {
        matrix_fragment_12 = matrix_fragment_12 + matrix_fragment_15;
    }

    if (condition_5) {
        matrix_fragment_13 = matrix_fragment_14 * 0x4u + matrix_fragment_89;
    }

    if (!condition_2) {
        matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_2) {
        matrix_fragment_16 = matrix_fragment_16 * 0x4u + matrix_fragment_89;
    }

    if (!condition_6) {
        matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_6) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];

        matrix_fragment_12 = matrix_fragment_12 * 0x4u + matrix_fragment_89;
    }

    if (condition_1) {
        matrix_fragment_10 = matrix_fragment_17 + index_word_0;
    }

    if (condition_2) {
        matrix_fragment_15 = matrix_fragment_16 * 0x80u;
    }

    index_word_0 = threadIdx.x & 31u;

    if (condition_5) {
        matrix_fragment_13 = matrix_fragment_13 * 0x80u;
    }

    if (!condition_5) {
        matrix_fragment_19 = permute_signed(matrix_fragment_19, matrix_fragment_19, 0x5410u);
    }

    if (condition_1) {
        matrix_fragment_10 = matrix_fragment_10 * 0x4u + matrix_fragment_89;

        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (!condition_2) {
        matrix_fragment_23 = permute_signed(matrix_fragment_23, matrix_fragment_23, 0x5410u);
    }

    if (!condition_6) {
        matrix_fragment_27 = permute_signed(matrix_fragment_27, matrix_fragment_27, 0x5410u);
    }

    if (condition_1) {
        matrix_fragment_11 = matrix_fragment_10 * 0x80u;
    }

    if (!condition_5) {
        matrix_fragment_16 = matrix_fragment_19;

        matrix_fragment_17 = matrix_fragment_19;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_1) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_5) {
        matrix_fragment_18 = matrix_fragment_19;
    }

    if (!condition_2) {
        matrix_fragment_20 = matrix_fragment_23;
    }

    if (!condition_1) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (!condition_2) {
        matrix_fragment_21 = matrix_fragment_23;

        matrix_fragment_22 = matrix_fragment_23;
    }

    if (!condition_6) {
        matrix_fragment_24 = matrix_fragment_27;
    }

    if (!condition_1) {
        matrix_fragment_13 = matrix_fragment_15;

        matrix_fragment_14 = matrix_fragment_15;
    }

    if (!condition_6) {
        matrix_fragment_25 = matrix_fragment_27;

        matrix_fragment_26 = matrix_fragment_27;
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (condition_6) {
        matrix_fragment_11 = matrix_fragment_12 * 0x80u;
    }

    if (!condition_1) {
        matrix_fragment_12 = matrix_fragment_15;
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    __syncthreads();

    if (condition_6) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_5) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_6) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (condition_5) {
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

    if (condition_6) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);
    }

    uniform_index_word_5 = 0u;

    uniform_index_word_4 = 0x400u;

    uniform_parameter_word_6 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_7 = parameters[parameter_trained_weights_high_5];

    index_word_142 = matrix_fragment_89 * 0x20u;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_accumulator_64 = 0;
    matrix_accumulator_65 = 0;

    matrix_fragment_80 = 0;
    matrix_fragment_81 = 0;

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_fragment_68 = 0;
    matrix_fragment_69 = 0;

    matrix_fragment_72 = 0;
    matrix_fragment_73 = 0;

    matrix_fragment_84 = 0;
    matrix_fragment_85 = 0;

    matrix_fragment_70 = 0;
    matrix_fragment_71 = 0;

    matrix_fragment_74 = 0;
    matrix_fragment_75 = 0;

    matrix_accumulator_78 = 0;
    matrix_accumulator_79 = 0;

    matrix_accumulator_82 = 0;
    matrix_accumulator_83 = 0;

    matrix_fragment_76 = 0;
    matrix_fragment_77 = 0;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_66 = 0;
    matrix_accumulator_67 = 0;

    split_address_words(matrix_fragment_86, matrix_fragment_87,
                        combine_words(uniform_parameter_word_6, uniform_parameter_word_7));

    uniform_index_word_5 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

    uniform_index_word_4 = 0u;

    tile_word_143 = (matrix_fragment_89 << 0x9u) + uniform_index_word_5;

    tile_word_143 = index_word_0 * 0x10u + tile_word_143;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_16,
                                         matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_20,
                                         matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            uint64_t(uint32_t(matrix_fragment_89)) * uint64_t(uint32_t(0x4000u)) +
                                combine_words(matrix_fragment_86, matrix_fragment_87));

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_20, matrix_fragment_21),
                                            matrix_fragment_44, matrix_fragment_45,
                                            matrix_fragment_46, matrix_fragment_47, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x200ull, matrix_accumulator_48,
            matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x1000ull,
            matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x1200ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x2000ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x2200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x3000ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21) + 0x3200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        matrix_fragment_88 = (index_word_0 << 0x4u) + uniform_index_word_5;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x800u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_88 + 0x1800u,
                                            matrix_fragment_52, matrix_fragment_53,
                                            matrix_accumulator_54, matrix_accumulator_55,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x200u, matrix_fragment_20, matrix_fragment_21,
            matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1a00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0xa00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_100, matrix_fragment_101, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_accumulator_48, matrix_accumulator_49, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1200u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_48, matrix_accumulator_49, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_48, matrix_accumulator_49, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_48, matrix_accumulator_49, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_20,
                                 matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                 matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_100,
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
                            uint64_t(uint32_t(matrix_fragment_89)) * uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_fragment_86, matrix_fragment_87));

        split_address_words(matrix_fragment_100, matrix_fragment_101,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_accumulator_36, matrix_accumulator_37,
                                 matrix_accumulator_116, matrix_accumulator_117);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_100, matrix_fragment_101) + 0x10000ull,
            matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_94,
                                 matrix_fragment_95);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_50,
                                 matrix_fragment_51);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_100, matrix_fragment_101) + 0x10200ull,
            matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_36, matrix_accumulator_37,
                                 matrix_accumulator_108, matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_44,
                                 matrix_fragment_45);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x400u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_accumulator_36, matrix_accumulator_37,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_96,
                                 matrix_fragment_97);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1600u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1400u, matrix_accumulator_36, matrix_accumulator_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0xe00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_8, matrix_fragment_9, matrix_fragment_98,
                                 matrix_fragment_99);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x600u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1c00u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_accumulator_36,
                                 matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_54,
                                 matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_102,
                                 matrix_fragment_103);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119,
                                 matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_38,
                                 matrix_fragment_39, matrix_fragment_8, matrix_fragment_9,
                                 matrix_fragment_52, matrix_fragment_53);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_38,
                                 matrix_fragment_39, matrix_fragment_10, matrix_fragment_11,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_4, matrix_fragment_5, matrix_fragment_94,
                                 matrix_fragment_95);

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37,
                                 matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_38,
                                 matrix_fragment_39, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_92, matrix_fragment_93);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_92 = 0x2b28u;

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_54), 0x44004400u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_55), 0x44004400u);

        matrix_fragment_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_104,
                                 matrix_accumulator_105);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_94),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94),
            select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_98,
                                 matrix_fragment_99);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_88 + 0x1e00u, matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_106,
                                 matrix_accumulator_107);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_93 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_fragment_93 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93), 0xc400c400u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_93),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_110,
                                 matrix_accumulator_111);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93),
            select_half_operand<2, false, false>(matrix_fragment_88), 0x3b283b28u);

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_88));

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_36,
                                 matrix_accumulator_37);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

        matrix_fragment_30 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

        matrix_fragment_29 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

        matrix_fragment_28 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_88 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

        matrix_fragment_30 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_28),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28),
            select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_48,
                                 matrix_accumulator_49);

        matrix_fragment_28 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_16 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0x44004400u);

        matrix_fragment_17 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

        matrix_fragment_16 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_28),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_16 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28),
            select_half_operand<2, false, false>(matrix_fragment_16), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_18, matrix_fragment_19, matrix_fragment_96,
                                 matrix_fragment_97);

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_fragment_94));

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_16 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_30),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        matrix_fragment_19 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_18 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

        matrix_accumulator_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                                select_half_operand<2, false, false>(matrix_fragment_95));

        matrix_fragment_16 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_95 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_88),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0xc400c400u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30),
            select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_30 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11), 0x44004400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_17),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_19 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

        matrix_fragment_18 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_16),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0xc400c400u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_94),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_30 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88),
            select_half_operand<2, false, false>(matrix_fragment_93), 0x3b283b28u);

        matrix_fragment_88 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_16 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16),
            select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

        matrix_fragment_18 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94),
            select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

        matrix_fragment_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0xc400c400u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_30),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                                 matrix_fragment_9);

        matrix_fragment_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_18 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

        matrix_accumulator_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_19),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_8 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

        matrix_fragment_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_fragment_95));

        matrix_fragment_95 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30),
            select_half_operand<2, false, false>(matrix_fragment_88), 0x3b283b28u);

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_19 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_100 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_18),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_95 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_94),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_19 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        matrix_fragment_29 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_36), 0x44004400u);

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_100 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_100), 0xc400c400u);

        matrix_fragment_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_48), 0x44004400u);

        matrix_fragment_18 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_49), 0x44004400u);

        matrix_fragment_17 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94),
            select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

        matrix_fragment_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_31 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_19),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_18 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

        matrix_fragment_101 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_100),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_17 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_34 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_37), 0x44004400u);

        matrix_fragment_29 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19),
            select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

        matrix_fragment_19 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_101 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_100),
            select_half_operand<2, false, false>(matrix_fragment_101), 0x3b283b28u);

        matrix_fragment_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_35 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0xc400c400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_31),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_101));

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_4 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_19),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31),
            select_half_operand<2, false, false>(matrix_fragment_34), 0x3b283b28u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_16 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19),
            select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_11 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                select_half_operand<2, false, false>(matrix_fragment_88));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                                select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_35),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_accumulator_36 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35),
            select_half_operand<2, false, false>(matrix_fragment_88), 0x3b283b28u);

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                          select_half_operand<2, false, false>(matrix_accumulator_36));

        matrix_fragment_33 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_17 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0x44004400u);

        matrix_fragment_19 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0x44004400u);

        matrix_accumulator_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                                select_half_operand<2, false, false>(matrix_fragment_88));

        matrix_fragment_35 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

        matrix_fragment_19 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

        matrix_accumulator_37 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_36), 0xc400c400u);

        matrix_accumulator_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_35),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_17 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_19),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_37),
            select_half_operand<0, false, true>(matrix_fragment_92), 0x37283728u);

        matrix_accumulator_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35),
            select_half_operand<2, false, false>(matrix_accumulator_36), 0x3b283b28u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_37),
            select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

        matrix_fragment_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_17), 0x3b283b28u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19),
            select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_accumulator_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_accumulator_36));

        matrix_fragment_92 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_92));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_18 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_29 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_14 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                          select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_19 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                          select_half_operand<2, false, false>(matrix_fragment_88));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                          select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                          select_half_operand<2, false, false>(matrix_fragment_92));

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_66,
                                 matrix_accumulator_67);

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_6 = uniform_index_word_4 + 0x20u;

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_84,
                                 matrix_fragment_85);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_80,
                                 matrix_fragment_81);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_78,
                                 matrix_accumulator_79);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_72,
                                 matrix_fragment_73);

        multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_64,
                                 matrix_accumulator_65);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_76,
                                 matrix_fragment_77);

        multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_74,
                                 matrix_fragment_75);

        multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_26, matrix_fragment_27, matrix_fragment_70,
                                 matrix_fragment_71);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_24, matrix_fragment_25, matrix_fragment_68,
                                 matrix_fragment_69);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_2,
                                 matrix_accumulator_3);

        split_address_words(matrix_fragment_86, matrix_fragment_87,
                            combine_words(matrix_fragment_86, matrix_fragment_87) + 0x400u);

        uniform_index_word_4 = uniform_parameter_word_6;
    } while (!uniform_condition_0);
    address_word_146 = parameters[parameter_trained_weights_low_4];
    address_word_147 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_6 = index_word_142 + 0x10u;

    index_word_141 = index_word_0;

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = index_word_142 + 0x18u;

    matrix_fragment_4 = index_word_142 + 0x8u;

    matrix_fragment_11 = index_word_141 & 0xfffffffcu;

    matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_6) >> 0x1u);

    matrix_fragment_4 = uint32_t(uint32_t(matrix_fragment_4) >> 0x1u);

    matrix_fragment_9 = matrix_fragment_6 + index_word_0 + (0u - matrix_fragment_11);

    matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_7) >> 0x1u);

    matrix_fragment_7 = index_word_141 & 0x7ffffffcu;

    matrix_fragment_5 = matrix_fragment_4 + index_word_0 + (0u - matrix_fragment_11);

    matrix_fragment_7 = 0u - matrix_fragment_7 + index_word_0;

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_11 = matrix_fragment_6 + index_word_0 + (0u - matrix_fragment_11);

    matrix_fragment_7 = matrix_fragment_7 * 0x2u + index_word_142;

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x18010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    address_word_140 = uint32_t(int32_t(matrix_fragment_7) >> 0x1u);

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x18010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(address_word_140)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x18010ull, memory_ranges,
        error_counts);

    matrix_fragment_38 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x18010ull, memory_ranges,
        error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                      select_half_operand<2, false, false>(0u));

    uniform_index_word_4 = 0u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                       select_half_operand<2, false, false>(0u));

    uniform_parameter_word_6 = uniform_index_word_5;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                      select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                      select_half_operand<2, false, false>(matrix_fragment_31));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_84,
                                         matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_36 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_143 + 0x1800u, matrix_fragment_80, matrix_fragment_81,
        matrix_accumulator_82, matrix_accumulator_83, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_12);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_22);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_23);

    matrix_accumulator_37 = unpack_fp8_bits(matrix_fragment_19);

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

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_fragment_89)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x14200u);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
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
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5) - 0x200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                            matrix_accumulator_36, matrix_accumulator_37,
                                            matrix_fragment_38, matrix_fragment_39, memory_ranges,
                                            error_counts);

        matrix_fragment_46 = (index_word_0 << 0x4u) + uniform_parameter_word_6;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x1000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_46, matrix_accumulator_48, matrix_accumulator_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_6 = uniform_parameter_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_46 + 0x800u,
                                            matrix_fragment_52, matrix_fragment_53,
                                            matrix_accumulator_54, matrix_accumulator_55,
                                            shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_46 + 0x1000u,
                                            matrix_accumulator_56, matrix_accumulator_57,
                                            matrix_accumulator_58, matrix_accumulator_59,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_46 + 0x1800u,
                                            matrix_accumulator_60, matrix_accumulator_61,
                                            matrix_accumulator_62, matrix_accumulator_63,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_accumulator_48,
                                 matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_accumulator_48,
                                 matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_accumulator_48,
                                 matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_accumulator_48,
                                 matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_52,
                                 matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_32, matrix_fragment_33,
                                 matrix_fragment_6, matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_fragment_34, matrix_fragment_35,
                                 matrix_accumulator_2, matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_accumulator_56,
                                 matrix_accumulator_57, matrix_accumulator_58,
                                 matrix_accumulator_59, matrix_accumulator_36,
                                 matrix_accumulator_37, matrix_fragment_44, matrix_fragment_45);

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
                                 matrix_accumulator_63, matrix_accumulator_36,
                                 matrix_accumulator_37, matrix_fragment_10, matrix_fragment_11);

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
                        uint64_t(uint32_t(matrix_fragment_89)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_146, address_word_147));

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

    split_address_words(address_word_144, address_word_145,
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

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_28,
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

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_48 = 0;
    matrix_accumulator_49 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                         select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1000u, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

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

    matrix_accumulator_36 = 0;
    matrix_accumulator_37 = 0;

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

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x1800u, matrix_fragment_8,
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

    matrix_accumulator_138 = 0;
    matrix_accumulator_139 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    packed_half_132 = 0;
    packed_half_133 = 0;

    uniform_index_word_4 = 0u;

    uniform_parameter_word_6 = uniform_index_word_5;

    __syncthreads();
    do {
        split_address_words(address_word_148, packed_half_149,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_144, address_word_145));

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0xa00ull, matrix_accumulator_64,
            matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x800ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x600ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x400ull, matrix_fragment_76,
            matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_148, packed_half_149) - 0x200ull, matrix_fragment_80,
            matrix_fragment_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(address_word_148, packed_half_149),
                                            matrix_fragment_84, matrix_fragment_85,
                                            matrix_fragment_86, matrix_fragment_87, memory_ranges,
                                            error_counts);

        packed_half_150 = (index_word_0 << 0x4u) + uniform_parameter_word_6;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(address_word_144, address_word_145,
                            combine_words(address_word_144, address_word_145) + 0x3000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_6 = uniform_parameter_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150 + 0x800u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150 + 0x1000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_150 + 0x1800u, matrix_fragment_100, matrix_fragment_101,
            matrix_fragment_102, matrix_fragment_103, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(packed_half_132, packed_half_133, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_64, matrix_accumulator_65, packed_half_132,
                                 packed_half_133);

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

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_66, matrix_accumulator_67,
                                 matrix_accumulator_130, matrix_accumulator_131);

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

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_68, matrix_fragment_69,
                                 matrix_accumulator_108, matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_70, matrix_fragment_71,
                                 matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_48,
                                 matrix_accumulator_49);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_72, matrix_fragment_73,
                                 matrix_accumulator_112, matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_74, matrix_fragment_75,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_fragment_76, matrix_fragment_77,
                                 matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_78, matrix_accumulator_79,
                                 matrix_accumulator_54, matrix_accumulator_55);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_78, matrix_accumulator_79,
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
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_82, matrix_accumulator_83,
                                 matrix_accumulator_58, matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_fragment_100, matrix_fragment_101, matrix_fragment_102,
                                 matrix_fragment_103, matrix_accumulator_82, matrix_accumulator_83,
                                 matrix_accumulator_122, matrix_accumulator_123);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_36,
                                 matrix_accumulator_37);

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

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    condition_1 = (index_word_0 & 0x1u) != 0;

    condition_0 = (index_word_0 & 0x2u) != 0;

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_11);

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_35);

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_33 = transpose_half_tile(matrix_fragment_33);

    matrix_fragment_39 = transpose_half_tile(matrix_fragment_39);

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        uint64_t(uint32_t(matrix_fragment_100)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_fragment_32 = transpose_half_tile(matrix_fragment_32);

    matrix_fragment_38 = transpose_half_tile(matrix_fragment_38);

    matrix_accumulator_121 = transpose_half_tile(matrix_accumulator_121);

    matrix_accumulator_59 = transpose_half_tile(matrix_accumulator_59);

    matrix_accumulator_125 = transpose_half_tile(matrix_accumulator_125);

    matrix_accumulator_123 = transpose_half_tile(matrix_accumulator_123);

    matrix_accumulator_57 = transpose_half_tile(matrix_accumulator_57);

    matrix_accumulator_61 = transpose_half_tile(matrix_accumulator_61);

    matrix_accumulator_63 = transpose_half_tile(matrix_accumulator_63);

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    uniform_index_word_4 = blockIdx.y;

    uniform_parameter_word_6 = parameters[parameter_optional_control_10];
    uniform_parameter_word_7 = parameters[parameter_scalar_control_11];

    matrix_fragment_101 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0x2c120ull, memory_ranges,
        error_counts);

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(packed_half_133),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    address_word_145 = matrix_fragment_100 * 0x800u;

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    split_address_words(address_word_144, address_word_145,
                        uint64_t(uint32_t(address_word_145)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

    split_address_words(address_word_144, address_word_145,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_144, address_word_145));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    __syncthreads();

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    address_word_148 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
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

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_132),
                                      select_half_operand<2, false, false>(packed_half_132),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x24120ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    packed_half_153 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x24320ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_fragment_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

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

    matrix_fragment_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                      select_half_operand<2, false, false>(matrix_accumulator_48),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                      select_half_operand<2, false, false>(matrix_accumulator_49),
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

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
                                     select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_102 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_accumulator_82 =
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
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_102 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                       select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_79 =
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
        matrix_accumulator_83 = result_word;
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

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_accumulator_79));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_94), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_103 = result_word;
    }

    matrix_accumulator_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_102 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

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
        matrix_accumulator_82 = result_word;
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
        combine_words(address_word_144, address_word_145) + 0x24520ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    packed_half_155 = permute_signed(packed_half_152, packed_half_152, 0x5432u);

    packed_half_154 = permute_signed(matrix_fragment_95, matrix_fragment_95, 0x5432u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x24720ull, matrix_fragment_76,
        matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                     select_half_operand<2, false, false>(packed_half_155));

    packed_half_155 = permute_signed(matrix_fragment_94, matrix_fragment_94, 0x5432u);

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(matrix_accumulator_82));

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
        combine_words(address_word_144, address_word_145) + 0x24920ull, matrix_fragment_80,
        matrix_fragment_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x24b20ull, matrix_fragment_84,
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
        combine_words(address_word_144, address_word_145) + 0x24d20ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_103 = half_to_float_bits((matrix_fragment_95 & 65535u) * 65537u);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                     select_half_operand<2, false, false>(matrix_fragment_92));

    packed_half_156 = half_to_float_bits((matrix_fragment_95 >> 16) * 65537u);

    packed_half_155 = half_to_float_bits((matrix_fragment_94 & 65535u) * 65537u);

    float_bits_158 = half_to_float_bits((matrix_fragment_94 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x24f20ull, matrix_fragment_92,
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

    matrix_fragment_103 = rsqrt_bits(matrix_fragment_103);

    packed_half_156 = rsqrt_bits(packed_half_156);

    packed_half_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_152), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_159 = result_word;
    }

    packed_half_154 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_164 = result_word;
    }

    packed_half_153 = half_to_float_bits((packed_half_154 & 65535u) * 65537u);

    packed_half_154 = half_to_float_bits((packed_half_154 >> 16) * 65537u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_99 = half_to_float_bits((packed_half_152 & 65535u) * 65537u);

    packed_half_152 = half_to_float_bits((packed_half_152 >> 16) * 65537u);

    packed_half_156 = pack_half(packed_half_156, matrix_fragment_103);

    packed_half_155 = rsqrt_bits(packed_half_155);

    matrix_fragment_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_99 = rsqrt_bits(matrix_fragment_99);

    matrix_accumulator_131 = permute_signed(matrix_fragment_98, matrix_fragment_98, 0x5432u);

    packed_half_152 = rsqrt_bits(packed_half_152);

    float_bits_158 = rsqrt_bits(float_bits_158);

    packed_half_153 = rsqrt_bits(packed_half_153);

    packed_half_154 = rsqrt_bits(packed_half_154);

    matrix_fragment_98 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_accumulator_131 = permute_signed(matrix_fragment_102, matrix_fragment_102, 0x5432u);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(packed_half_159));

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(packed_half_164));

    matrix_fragment_102 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                       select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_99 = pack_half(packed_half_152, matrix_fragment_99);

    matrix_accumulator_131 = permute_signed(matrix_fragment_96, matrix_fragment_96, 0x5432u);

    packed_half_155 = pack_half(float_bits_158, packed_half_155);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_151), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    packed_half_132 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_132),
                                          select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_152 = pack_half(packed_half_154, packed_half_153);

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0x4100410u);

    packed_half_154 = permute_signed(matrix_fragment_97, matrix_fragment_97, 0x5432u);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_131 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                             select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_4 = half_to_float_bits((packed_half_153 & 65535u) * 65537u);

    packed_half_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(packed_half_154));

    packed_half_154 = half_to_float_bits((packed_half_153 >> 16) * 65537u);

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_101 = pack_half(0u, matrix_fragment_101);

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                          select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_133),
                                        select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(packed_half_152));

    packed_half_133 = rsqrt_bits(packed_half_154);

    packed_half_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_132),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    packed_half_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_102 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_132),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x4100410u);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x4100410u);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_98 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_152));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_149), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_accumulator_138 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    packed_half_132 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                    select_half_operand<2, false, false>(0u));

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                          select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_5 = half_to_float_bits((packed_half_153 >> 16) * 65537u);

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_136 = half_to_float_bits((packed_half_153 & 65535u) * 65537u);

    matrix_fragment_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                      select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                       select_half_operand<2, false, false>(packed_half_132));

    packed_half_133 = pack_half(packed_half_133, matrix_fragment_4);

    packed_half_132 = rsqrt_bits(matrix_accumulator_136);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                      select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_150), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_157 = result_word;
    }

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_accumulator_136 = permute_signed(packed_half_151, packed_half_151, 0x5432u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_4 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x4100410u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_133));

    packed_half_151 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                     select_half_operand<2, false, false>(matrix_accumulator_136));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_148), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_133 = result_word;
    }

    matrix_accumulator_131 = half_to_float_bits((matrix_fragment_4 & 65535u) * 65537u);

    matrix_fragment_26 = half_to_float_bits((matrix_fragment_4 >> 16) * 65537u);

    packed_half_151 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_151), 0x4100410u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_131 = rsqrt_bits(matrix_accumulator_131);

    matrix_fragment_24 = half_to_float_bits((packed_half_151 & 65535u) * 65537u);

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(packed_half_160));

    packed_half_151 = half_to_float_bits((packed_half_151 >> 16) * 65537u);

    matrix_accumulator_135 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = rsqrt_bits(matrix_fragment_26);

    packed_half_132 = pack_half(matrix_fragment_5, packed_half_132);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_28 = permute_signed(packed_half_149, packed_half_149, 0x5432u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_132));

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_24);

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(packed_half_132));

    packed_half_151 = rsqrt_bits(packed_half_151);

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_132));

    matrix_fragment_27 = permute_signed(packed_half_150, packed_half_150, 0x5432u);

    packed_half_133 = add_half_pairs(select_half_operand<2, false, false>(address_word_148),
                                     select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 = half_to_float_bits((matrix_fragment_6 & 65535u) * 65537u);

    matrix_fragment_31 = half_to_float_bits((matrix_fragment_6 >> 16) * 65537u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                     select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_25 = pack_half(matrix_fragment_26, matrix_accumulator_131);

    matrix_fragment_26 = permute_signed(packed_half_133, packed_half_133, 0x5432u);

    packed_half_150 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_150), 0x4100410u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_151 = pack_half(packed_half_151, matrix_fragment_24);

    packed_half_133 = add_half_pairs(select_half_operand<2, false, false>(packed_half_133),
                                     select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(packed_half_132));

    matrix_fragment_29 = half_to_float_bits((packed_half_150 & 65535u) * 65537u);

    packed_half_133 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_133), 0x4100410u);

    matrix_accumulator_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(packed_half_151));

    packed_half_149 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_149), 0x4100410u);

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_48 = rsqrt_bits(matrix_fragment_29);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = half_to_float_bits((packed_half_149 & 65535u) * 65537u);

    packed_half_149 = half_to_float_bits((packed_half_149 >> 16) * 65537u);

    matrix_fragment_29 = half_to_float_bits((packed_half_133 & 65535u) * 65537u);

    packed_half_133 = half_to_float_bits((packed_half_133 >> 16) * 65537u);

    matrix_accumulator_130 = rsqrt_bits(matrix_accumulator_130);

    packed_half_150 = half_to_float_bits((packed_half_150 >> 16) * 65537u);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_31 = rsqrt_bits(matrix_fragment_31);

    matrix_fragment_30 = rsqrt_bits(matrix_fragment_29);

    packed_half_133 = rsqrt_bits(packed_half_133);

    matrix_fragment_28 = rsqrt_bits(matrix_fragment_28);

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_accumulator_49 = rsqrt_bits(packed_half_150);

    matrix_accumulator_130 = pack_half(matrix_fragment_31, matrix_accumulator_130);

    matrix_accumulator_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_133 = pack_half(packed_half_133, matrix_fragment_30);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    packed_half_149 = pack_half(packed_half_149, matrix_fragment_28);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_48 = pack_half(matrix_accumulator_49, matrix_accumulator_48);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_accumulator_54));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_accumulator_48 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(packed_half_133));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_accumulator_49));

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_78,
                             matrix_accumulator_79);

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

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_accumulator_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_22 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_23 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_98 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_97 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_114 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_113 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_78 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3c203c20u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_54 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_55 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_48 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_49 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_51 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_78 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3c203c20u);

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    matrix_accumulator_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_fragment_81 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3e473e47u);

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_51 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                       select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_73 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_72 = (matrix_accumulator_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_79 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3c203c20u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_53 = (matrix_fragment_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_78 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_fragment_50 = (matrix_fragment_85 << 0x5u) + 0x7ff88000u;

    matrix_fragment_81 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_52 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_51 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_83 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_accumulator_79 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3e473e47u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_accumulator_64 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_67 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_fragment_74 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_75 = (matrix_accumulator_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_65 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_66 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_68 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_71 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_69 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                       select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_70 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 = index_word_0 * 0x4u;

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_77 = matrix_fragment_19 & 0x1cu;

    matrix_fragment_19 = condition_1 ? matrix_fragment_17 : matrix_fragment_16;

    matrix_fragment_20 = condition_1 ? matrix_fragment_21 : matrix_fragment_18;

    matrix_fragment_16 = condition_1 ? matrix_fragment_16 : matrix_fragment_17;

    matrix_fragment_21 = condition_1 ? matrix_fragment_18 : matrix_fragment_21;

    matrix_fragment_94 =
        uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1du) >> 32) + matrix_fragment_77;

    matrix_fragment_17 = !condition_0 ? matrix_fragment_19 : matrix_fragment_20;

    matrix_fragment_77 = !condition_0 ? matrix_fragment_20 : matrix_fragment_19;

    matrix_fragment_18 = !condition_0 ? matrix_fragment_16 : matrix_fragment_21;

    matrix_fragment_16 = !condition_0 ? matrix_fragment_21 : matrix_fragment_16;

    matrix_fragment_93 = 0x1u ^ matrix_fragment_94;

    matrix_fragment_92 = 0x2u ^ matrix_fragment_94;

    matrix_fragment_19 = 0x3u ^ matrix_fragment_94;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_18), "r"(matrix_fragment_93), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(matrix_fragment_92), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(matrix_fragment_19), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    condition_0 = (index_word_0 & 0x8u) != 0;

    condition_1 = (index_word_0 & 0x10u) != 0;

    matrix_fragment_20 = !condition_0 ? matrix_fragment_17 : matrix_fragment_18;

    matrix_fragment_21 = !condition_0 ? matrix_fragment_18 : matrix_fragment_17;

    matrix_accumulator_79 = !condition_0 ? matrix_fragment_77 : matrix_fragment_16;

    matrix_fragment_80 = !condition_0 ? matrix_fragment_16 : matrix_fragment_77;

    matrix_accumulator_82 = !condition_1 ? matrix_fragment_20 : matrix_accumulator_79;

    matrix_fragment_17 = !condition_1 ? matrix_fragment_21 : matrix_fragment_80;

    matrix_fragment_20 = !condition_1 ? matrix_accumulator_79 : matrix_fragment_20;

    matrix_fragment_21 = !condition_1 ? matrix_fragment_80 : matrix_fragment_21;

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_17 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_17),
                                        select_half_operand<1, false, false>(matrix_fragment_17));

    matrix_fragment_17 = permute_signed(matrix_fragment_17, matrix_fragment_17, 0x5410u);

    matrix_fragment_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x4100410u);

    matrix_fragment_20 = half_to_float_bits((matrix_fragment_17 & 65535u) * 65537u);

    matrix_fragment_21 = half_to_float_bits((matrix_fragment_17 >> 16) * 65537u);

    matrix_fragment_16 = uint32_t(int32_t(index_word_141) >> 0x2u);

    matrix_fragment_20 = reciprocal_bits(matrix_fragment_20);

    matrix_fragment_18 = matrix_fragment_16 + 0x8u;

    matrix_fragment_77 = uint32_t(int32_t(index_word_141) >> 0x1fu);

    matrix_fragment_21 = reciprocal_bits(matrix_fragment_21);

    matrix_fragment_17 = uint32_t(int32_t(matrix_fragment_18) >> 0x1fu);

    matrix_fragment_77 =
        uint32_t(combine_words(matrix_fragment_77, 0u) << 0x5u >> 32) + matrix_fragment_16;

    matrix_accumulator_79 =
        uint32_t(combine_words(matrix_fragment_17, 0u) << 0x5u >> 32) + matrix_fragment_18;

    matrix_fragment_77 = matrix_fragment_77 & 0xffffffe0u;

    matrix_accumulator_79 = matrix_accumulator_79 & 0xffffffe0u;

    matrix_fragment_95 = matrix_fragment_16 + (0u - matrix_fragment_77);

    matrix_fragment_87 = matrix_fragment_18 + (0u - matrix_accumulator_79);

    matrix_fragment_84 = pack_half(matrix_fragment_21, matrix_fragment_20);

    matrix_fragment_20 = transpose_half_tile(matrix_fragment_10);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_95), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_87), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_accumulator_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = transpose_half_tile(matrix_fragment_34);

    matrix_fragment_88 = uint32_t(int64_t(int32_t(index_word_141)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_90 = uint32_t(int64_t(int32_t(index_word_141)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_89 = uint32_t(int32_t(matrix_fragment_88) >> 0x1fu);

    matrix_fragment_91 = uint32_t(int32_t(matrix_fragment_90) >> 0x1fu);

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_79 = permute_signed(matrix_fragment_85, matrix_fragment_85, 0x5410u);

    matrix_fragment_89 =
        uint32_t(combine_words(matrix_fragment_89, 0u) << 0x5u >> 32) + matrix_fragment_88;

    matrix_fragment_77 = permute_signed(matrix_fragment_87, matrix_fragment_87, 0x5410u);

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_91 =
        uint32_t(combine_words(matrix_fragment_91, 0u) << 0x5u >> 32) + matrix_fragment_90;

    matrix_fragment_80 = transpose_half_tile(matrix_fragment_8);

    matrix_fragment_89 = matrix_fragment_89 & 0xffffffe0u;

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_91 = matrix_fragment_91 & 0xffffffe0u;

    matrix_accumulator_82 = transpose_half_tile(matrix_fragment_9);

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_89 = matrix_fragment_88 + (0u - matrix_fragment_89);

    matrix_fragment_91 = matrix_fragment_90 + (0u - matrix_fragment_91);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_23 = transpose_half_tile(matrix_fragment_13);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_86 = transpose_half_tile(matrix_accumulator_37);

    matrix_fragment_22 = transpose_half_tile(matrix_fragment_12);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_89), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(matrix_fragment_91), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    matrix_fragment_85 = transpose_half_tile(matrix_accumulator_36);

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_87 = transpose_half_tile(matrix_accumulator_120);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_84 = transpose_half_tile(matrix_accumulator_58);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_80 = permute_signed(matrix_fragment_89, matrix_fragment_89, 0x5410u);

    matrix_accumulator_82 = permute_signed(matrix_fragment_91, matrix_fragment_91, 0x5410u);

    matrix_fragment_89 = transpose_half_tile(matrix_accumulator_124);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_85 = transpose_half_tile(matrix_accumulator_122);

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_86 = transpose_half_tile(matrix_accumulator_56);

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_88 = transpose_half_tile(matrix_accumulator_60);

    matrix_fragment_90 = transpose_half_tile(matrix_accumulator_62);

    matrix_fragment_91 = transpose_half_tile(matrix_accumulator_126);

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_55),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_67 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_66 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                      select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_97 = uint32_t(uint32_t(index_word_142) >> 0x1u);

    matrix_accumulator_117 = index_word_141 & 0xfffffffcu;

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_accumulator_113 = matrix_fragment_97 + 0x4u;

    matrix_fragment_99 = matrix_fragment_97 + 0x8u;

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_50,
                             matrix_fragment_51);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_48,
                             matrix_accumulator_49);

    matrix_fragment_97 = matrix_fragment_97 + 0xcu;

    matrix_fragment_9 = matrix_accumulator_113 + index_word_0 + (0u - matrix_accumulator_117);

    matrix_fragment_11 = matrix_fragment_99 + index_word_0 + (0u - matrix_accumulator_117);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_accumulator_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                          select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_accumulator_49 = matrix_fragment_97 + index_word_0 + (0u - matrix_accumulator_117);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_98 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_52, matrix_fragment_53,
                        int64_t(int32_t(address_word_140)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_48, matrix_accumulator_49,
                        uint64_t(uint32_t(matrix_accumulator_49)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_52, matrix_fragment_53) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_accumulator_112 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_48, matrix_accumulator_49) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                      select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_72,
                                         matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_143 + 0x800u, matrix_fragment_76, matrix_fragment_77,
        matrix_accumulator_78, matrix_accumulator_79, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_115 = matrix_fragment_100 * 0x100u;

    split_address_words(matrix_accumulator_114, matrix_accumulator_115,
                        uint64_t(uint32_t(matrix_accumulator_115)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_146, address_word_147));

    split_address_words(matrix_accumulator_114, matrix_accumulator_115,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_114, matrix_accumulator_115));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2c130ull,
        matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2c330ull,
        matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2d130ull,
        matrix_fragment_52, matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2d330ull,
        matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2e130ull,
        matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2e330ull,
        matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2f330ull,
        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2f130ull,
        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
        memory_ranges, error_counts);

    matrix_accumulator_116 = (index_word_0 << 0x4u) + uniform_index_word_5;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116, matrix_fragment_76, matrix_fragment_77,
        matrix_accumulator_78, matrix_accumulator_79, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_81 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    uniform_index_word_5 = blockIdx.x;

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_parameter_word_7;

    uniform_index_word_5 = (uniform_index_word_5 << 0x3u) + uniform_parameter_word_6;

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_80,
                             matrix_fragment_81);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0x200u, matrix_fragment_80, matrix_fragment_81,
        matrix_accumulator_82, matrix_accumulator_83, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_10,
                             matrix_fragment_11);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0x400u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                             matrix_fragment_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_80,
                             matrix_fragment_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_80,
                             matrix_fragment_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_80,
                             matrix_fragment_81, matrix_accumulator_82, matrix_accumulator_83,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_76,
                             matrix_fragment_77);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0x600u, matrix_fragment_76, matrix_fragment_77,
        matrix_accumulator_78, matrix_accumulator_79, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_80 = parameters[parameter_optional_control_20];
    matrix_fragment_81 = parameters[parameter_optional_control_21];

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_124,
                             matrix_accumulator_125);

    condition_2 = int32_t(matrix_fragment_80) > int32_t(0u);

    condition_3 = int32_t(matrix_fragment_81) > int32_t(0u);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_122,
                             matrix_accumulator_123);

    matrix_accumulator_124 = index_word_0;

    index_word_128 = condition_2 ? matrix_fragment_80 : index_word_128;

    index_word_129 = condition_3 ? matrix_fragment_81 : index_word_129;

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_fragment_8 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x2u >> 32) + matrix_fragment_16;

    matrix_accumulator_125 = matrix_fragment_8 & 0xfffffffcu;

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_120,
                             matrix_accumulator_121);

    matrix_fragment_80 = matrix_fragment_16 + uniform_index_word_5 + (0u - matrix_accumulator_125);

    matrix_accumulator_121 =
        uint32_t(int64_t(int32_t(matrix_accumulator_124)) * (1ll << 0x1cu) >> 32) +
        uniform_index_word_4;

    condition_3 = int32_t(matrix_fragment_80) >= int32_t(index_word_129);

    condition_2 = int32_t(matrix_accumulator_121) >= int32_t(index_word_128);

    condition_3 = int32_t(matrix_fragment_80) < int32_t(0u) || condition_3;

    condition_2 = int32_t(matrix_accumulator_121) < int32_t(0u) || condition_2;

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_126,
                             matrix_accumulator_127);

    condition_5 = condition_2 | condition_3;

    matrix_accumulator_127 = matrix_accumulator_121 + 0x2u;

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_130,
                             matrix_accumulator_131);

    condition_4 = int32_t(matrix_accumulator_127) >= int32_t(index_word_128);

    if (!condition_5) {
        matrix_accumulator_130 = parameters[parameter_output_low_2];
        matrix_accumulator_131 = parameters[parameter_output_high_3];
    }

    condition_4 = int32_t(matrix_accumulator_127) < int32_t(0u) || condition_4;

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_76,
                             matrix_fragment_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_122,
                             matrix_accumulator_123);

    matrix_fragment_81 = matrix_fragment_100 * 0x2u;

    condition_6 = condition_4 | condition_3;

    if (!condition_5) {
        matrix_accumulator_135 = matrix_fragment_80 * 0x4u;
    }

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_fragment_100 = index_word_128 * matrix_fragment_81 + matrix_accumulator_121;

        matrix_accumulator_122 = parameters[parameter_output_low_2];
        matrix_accumulator_123 = parameters[parameter_output_high_3];
    }

    matrix_fragment_76 = index_word_129 * 0x4u;

    if (!condition_5) {
        matrix_fragment_100 = matrix_fragment_76 * matrix_fragment_100 + matrix_accumulator_135;
    }

    packed_half_133 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                    select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_accumulator_124 = parameters[parameter_output_low_2];
        matrix_accumulator_125 = parameters[parameter_output_high_3];
    }

    if (!condition_5) {
        matrix_accumulator_135 = matrix_fragment_100 + index_word_0 + (0u - matrix_accumulator_117);

        address_word_147 = permute_signed(matrix_fragment_10, packed_half_133, 0x5410u);
    }

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(0u));

    packed_half_133 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                    select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        split_address_words(matrix_accumulator_130, matrix_accumulator_131,
                            int64_t(int32_t(matrix_accumulator_135)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_130, matrix_accumulator_131));
    }

    matrix_fragment_77 = matrix_fragment_81 + 0x1u;

    if (!condition_5) {
        matrix_accumulator_135 = permute_signed(matrix_fragment_10, packed_half_133, 0x5410u);
    }

    if (!condition_6) {
        packed_half_133 = permute_signed(matrix_fragment_11, matrix_fragment_8, 0x5410u);

        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];

        matrix_accumulator_120 = matrix_fragment_80 * 0x4u;

        matrix_accumulator_137 = index_word_128 * matrix_fragment_81 + matrix_accumulator_127;
    }

    if (!condition_5) {
        matrix_accumulator_126 = matrix_fragment_80 * 0x4u;

        matrix_accumulator_139 = index_word_128 * matrix_fragment_77 + matrix_accumulator_121;
    }

    if (!condition_6) {
        matrix_accumulator_137 =
            matrix_fragment_76 * matrix_accumulator_137 + matrix_accumulator_120;
    }

    if (!condition_5) {
        matrix_accumulator_139 =
            matrix_fragment_76 * matrix_accumulator_139 + matrix_accumulator_126;
    }

    if (!condition_6) {
        matrix_fragment_8 = matrix_fragment_80 * 0x4u;

        matrix_accumulator_83 = index_word_128 * matrix_fragment_77 + matrix_accumulator_127;

        matrix_accumulator_137 =
            matrix_accumulator_137 + index_word_0 + (0u - matrix_accumulator_117);
    }

    if (!condition_5) {
        matrix_accumulator_139 =
            matrix_accumulator_139 + index_word_0 + (0u - matrix_accumulator_117);
    }

    if (!condition_6) {
        matrix_accumulator_83 = matrix_fragment_76 * matrix_accumulator_83 + matrix_fragment_8;

        split_address_words(matrix_accumulator_124, matrix_accumulator_125,
                            int64_t(int32_t(matrix_accumulator_137)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_124, matrix_accumulator_125));
    }

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_accumulator_83 =
            matrix_accumulator_83 + index_word_0 + (0u - matrix_accumulator_117);
    }

    if (!condition_5) {
        split_address_words(matrix_accumulator_122, matrix_accumulator_123,
                            int64_t(int32_t(matrix_accumulator_139)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_122, matrix_accumulator_123));
    }

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_130, matrix_accumulator_131),
                                    address_word_147, memory_ranges, error_counts);
    }

    if (!condition_6) {
        split_address_words(matrix_accumulator_82, matrix_accumulator_83,
                            int64_t(int32_t(matrix_accumulator_83)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_124, matrix_accumulator_125),
                                    packed_half_133, memory_ranges, error_counts);
    }

    if (!condition_5) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_122, matrix_accumulator_123),
                                    matrix_accumulator_135, memory_ranges, error_counts);
    }

    if (!condition_6) {
        matrix_accumulator_123 = permute_signed(matrix_fragment_9, matrix_fragment_8, 0x5410u);
    }

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0x800u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_78 = unpack_fp8_bits(matrix_fragment_32);

    matrix_accumulator_79 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    if (!condition_6) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_82, matrix_accumulator_83),
                                    matrix_accumulator_123, memory_ranges, error_counts);
    }

    matrix_accumulator_82 = unpack_fp8_bits(matrix_fragment_34);

    matrix_accumulator_83 = unpack_fp8_bits(matrix_fragment_35);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_36, matrix_accumulator_37, matrix_accumulator_78,
                             matrix_accumulator_79);

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0xa00u, matrix_accumulator_36, matrix_accumulator_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0xc00u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_accumulator_36,
                             matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_116 + 0xe00u, matrix_fragment_32, matrix_fragment_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_80 = matrix_fragment_80 + 0x4u;

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_78,
                             matrix_accumulator_79);

    condition_5 = int32_t(matrix_fragment_80) >= int32_t(index_word_129);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_48,
                             matrix_accumulator_49);

    condition_5 = int32_t(matrix_fragment_80) < int32_t(0u) || condition_5;

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_36,
                             matrix_accumulator_37);

    condition_2 = condition_2 | condition_5;

    condition_4 = condition_4 | condition_5;

    if (!condition_2) {
        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_54,
                             matrix_accumulator_55);

    if (!condition_2) {
        matrix_fragment_39 = matrix_fragment_80 * 0x4u;
    }

    if (!condition_4) {
        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];

        matrix_fragment_68 = parameters[parameter_output_low_2];
        matrix_fragment_69 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_36,
                             matrix_accumulator_37);

    if (!condition_2) {
        matrix_fragment_34 = parameters[parameter_output_low_2];
        matrix_fragment_35 = parameters[parameter_output_high_3];

        matrix_accumulator_36 = index_word_128 * matrix_fragment_81 + matrix_accumulator_121;

        matrix_fragment_39 = matrix_fragment_76 * matrix_accumulator_36 + matrix_fragment_39;
    }

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_37 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_39 = matrix_fragment_39 + index_word_0 + (0u - matrix_accumulator_117);
    }

    if (!condition_4) {
        matrix_accumulator_49 = matrix_fragment_80 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_53 = permute_signed(matrix_accumulator_78, matrix_accumulator_37, 0x5410u);
    }

    if (!condition_4) {
        matrix_fragment_38 = index_word_128 * matrix_fragment_81 + matrix_accumulator_127;
    }

    if (!condition_2) {
        matrix_accumulator_48 = index_word_128 * matrix_fragment_77 + matrix_accumulator_121;

        matrix_fragment_51 = matrix_fragment_80 * 0x4u;

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    matrix_accumulator_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_fragment_39 = matrix_fragment_80 * 0x4u;

        matrix_fragment_32 = index_word_128 * matrix_fragment_77 + matrix_accumulator_127;

        matrix_fragment_38 = matrix_fragment_76 * matrix_fragment_38 + matrix_accumulator_49;
    }

    if (!condition_2) {
        matrix_accumulator_48 = matrix_fragment_76 * matrix_accumulator_48 + matrix_fragment_51;
    }

    if (!condition_4) {
        matrix_fragment_32 = matrix_fragment_76 * matrix_fragment_32 + matrix_fragment_39;

        matrix_accumulator_49 = matrix_fragment_38 + index_word_0 + (0u - matrix_accumulator_117);
    }

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_36 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_51 = matrix_accumulator_48 + index_word_0 + (0u - matrix_accumulator_117);
    }

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_accumulator_117 = matrix_fragment_32 + index_word_0 + (0u - matrix_accumulator_117);
    }

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_accumulator_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

        matrix_accumulator_79 =
            permute_signed(matrix_accumulator_79, matrix_accumulator_36, 0x5410u);
    }

    if (!condition_2) {
        matrix_accumulator_37 = permute_signed(matrix_fragment_72, matrix_accumulator_37, 0x5410u);

        split_address_words(matrix_fragment_34, matrix_fragment_35,
                            int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_34, matrix_fragment_35));
    }

    if (!condition_4) {
        matrix_fragment_73 = permute_signed(matrix_fragment_73, matrix_fragment_32, 0x5410u);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_53, memory_ranges, error_counts);
    }

    if (!condition_4) {
        split_address_words(matrix_fragment_68, matrix_fragment_69,
                            int64_t(int32_t(matrix_accumulator_117)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_68, matrix_fragment_69));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_accumulator_79, memory_ranges, error_counts);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_34, matrix_fragment_35),
                                    matrix_accumulator_37, memory_ranges, error_counts);
    }

    if (!condition_4) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_68, matrix_fragment_69),
                                    matrix_fragment_73, memory_ranges, error_counts);
    }

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25120ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25320ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25520ull, matrix_accumulator_36,
        matrix_accumulator_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25720ull, matrix_accumulator_48,
        matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25920ull, matrix_fragment_52,
        matrix_fragment_53, matrix_accumulator_54, matrix_accumulator_55, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25b20ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25d20ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_144, address_word_145) + 0x25f20ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_69 = result_word;
    }

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_71 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5432u);

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_69 = permute_signed(matrix_fragment_74, matrix_fragment_74, 0x5432u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x4100410u);

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_72 & 65535u) * 65537u);

    matrix_fragment_72 = half_to_float_bits((matrix_fragment_72 >> 16) * 65537u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x4100410u);

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

    matrix_fragment_75 = half_to_float_bits((matrix_fragment_74 & 65535u) * 65537u);

    matrix_fragment_74 = half_to_float_bits((matrix_fragment_74 >> 16) * 65537u);

    matrix_fragment_72 = rsqrt_bits(matrix_fragment_72);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_75 = rsqrt_bits(matrix_fragment_75);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_74 = rsqrt_bits(matrix_fragment_74);

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_72 = pack_half(matrix_fragment_72, matrix_fragment_73);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    matrix_fragment_74 = pack_half(matrix_fragment_74, matrix_fragment_75);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_8,
                             matrix_fragment_9);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = permute_signed(matrix_fragment_9, matrix_fragment_9, 0x5432u);

    matrix_fragment_11 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5432u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x4100410u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x4100410u);

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_9 & 65535u) * 65537u);

    matrix_fragment_9 = half_to_float_bits((matrix_fragment_9 >> 16) * 65537u);

    matrix_fragment_11 = half_to_float_bits((matrix_fragment_8 & 65535u) * 65537u);

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_8 >> 16) * 65537u);

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    matrix_fragment_9 = rsqrt_bits(matrix_fragment_9);

    matrix_fragment_11 = rsqrt_bits(matrix_fragment_11);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    matrix_fragment_10 = pack_half(matrix_fragment_9, matrix_fragment_10);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_68 = pack_half(matrix_fragment_8, matrix_fragment_11);

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(matrix_fragment_101));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_36,
                             matrix_accumulator_37);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_38,
                             matrix_fragment_39);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_33),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3c203c20u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_35),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3c203c20u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_accumulator_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3c203c20u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3e473e47u);

    matrix_accumulator_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_accumulator_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3c203c20u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_accumulator_54 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3c203c20u);

    matrix_accumulator_56 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3c203c20u);

    matrix_accumulator_58 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3c203c20u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3e473e47u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_accumulator_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_accumulator_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3e473e47u);

    matrix_accumulator_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_accumulator_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_54), 0x3e473e47u);

    matrix_accumulator_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_56), 0x3e473e47u);

    matrix_accumulator_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_58), 0x3e473e47u);

    matrix_fragment_4 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = (matrix_fragment_32 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = (matrix_fragment_34 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_36 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3c203c20u);

    matrix_accumulator_37 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3c203c20u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_27 = (matrix_fragment_33 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_5 = (matrix_fragment_45 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_6 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_28 = (matrix_fragment_35 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_2 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_3 = (matrix_accumulator_54 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = (matrix_accumulator_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_36 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_36), 0x3e473e47u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_37 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_37), 0x3e473e47u);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    matrix_accumulator_55 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3c203c20u);

    matrix_accumulator_57 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3c203c20u);

    matrix_accumulator_59 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3c203c20u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                        select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_55), 0x3e473e47u);

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                        select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_fragment_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                        select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_62 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3c203c20u);

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_57), 0x3e473e47u);

    matrix_accumulator_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_48),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_59), 0x3e473e47u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_8 = (matrix_accumulator_36 << 0x5u) + 0x7ff88000u;

    matrix_fragment_9 = (matrix_accumulator_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    matrix_fragment_10 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_49),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_accumulator_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3e473e47u);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_32 = (matrix_fragment_53 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_33 = (matrix_accumulator_55 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_34 = (matrix_accumulator_57 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<0, false, false>(matrix_fragment_96), 0x3d343d34u);

    matrix_fragment_35 = (matrix_accumulator_59 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_48 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_48), 0x3c203c20u);

    matrix_accumulator_37 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    matrix_accumulator_49 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_49), 0x3c203c20u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3c203c20u);

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_30 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_31 = (matrix_accumulator_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                       select_half_operand<2, false, false>(matrix_accumulator_36));

    matrix_accumulator_48 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_48), 0x3e473e47u);

    matrix_fragment_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    matrix_accumulator_36 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_37 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_49), 0x3e473e47u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3e473e47u);

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_43 = (matrix_accumulator_48 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_48 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                       select_half_operand<2, false, false>(matrix_accumulator_37));

    matrix_fragment_44 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_40 = (matrix_accumulator_49 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = (matrix_fragment_51 << 0x5u) + 0x7ff88000u;

    matrix_fragment_50 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_38 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_39 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                        select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_48 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    condition_2 = (index_word_0 & 0x1u) != 0;

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_47 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_accumulator_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                       select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_accumulator_48 = condition_2 ? matrix_fragment_47 : matrix_fragment_46;

    matrix_fragment_46 = condition_2 ? matrix_fragment_46 : matrix_fragment_47;

    condition_6 = (index_word_0 & 0x2u) != 0;

    matrix_fragment_47 = condition_2 ? matrix_fragment_50 : matrix_accumulator_49;

    matrix_accumulator_49 = condition_2 ? matrix_accumulator_49 : matrix_fragment_50;

    matrix_fragment_51 = !condition_6 ? matrix_accumulator_48 : matrix_fragment_47;

    matrix_fragment_50 = !condition_6 ? matrix_fragment_46 : matrix_accumulator_49;

    matrix_fragment_47 = !condition_6 ? matrix_fragment_47 : matrix_accumulator_48;

    matrix_fragment_46 = !condition_6 ? matrix_accumulator_49 : matrix_fragment_46;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_51), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(matrix_fragment_93), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(matrix_fragment_92), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_19), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_accumulator_48 = !condition_0 ? matrix_fragment_51 : matrix_fragment_50;

    matrix_accumulator_49 = !condition_0 ? matrix_fragment_50 : matrix_fragment_51;

    matrix_fragment_53 = !condition_0 ? matrix_fragment_47 : matrix_fragment_46;

    matrix_fragment_52 = !condition_0 ? matrix_fragment_46 : matrix_fragment_47;

    matrix_accumulator_54 = !condition_1 ? matrix_accumulator_48 : matrix_fragment_53;

    matrix_accumulator_55 = !condition_1 ? matrix_accumulator_49 : matrix_fragment_52;

    matrix_accumulator_48 = !condition_1 ? matrix_fragment_53 : matrix_accumulator_48;

    matrix_accumulator_49 = !condition_1 ? matrix_fragment_52 : matrix_accumulator_49;

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_55));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54),
                       select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_accumulator_54 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_54),
                       select_half_operand<1, false, false>(matrix_accumulator_54));

    matrix_accumulator_54 = permute_signed(matrix_accumulator_54, matrix_accumulator_54, 0x5410u);

    matrix_accumulator_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_54), 0x4100410u);

    matrix_fragment_19 = half_to_float_bits((matrix_accumulator_54 & 65535u) * 65537u);

    matrix_accumulator_54 = half_to_float_bits((matrix_accumulator_54 >> 16) * 65537u);

    matrix_fragment_19 = reciprocal_bits(matrix_fragment_19);

    matrix_fragment_17 =
        uint32_t(combine_words(matrix_fragment_17, 0u) << 0x5u >> 32) + matrix_fragment_18;

    matrix_accumulator_54 = reciprocal_bits(matrix_accumulator_54);

    matrix_fragment_17 = matrix_fragment_17 & 0xffffffe0u;

    matrix_fragment_47 = uint32_t(int64_t(int32_t(index_word_141)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_17 = matrix_fragment_18 + (0u - matrix_fragment_17);

    matrix_fragment_18 = uint32_t(int64_t(int32_t(index_word_141)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_accumulator_48 = uint32_t(int32_t(matrix_fragment_47) >> 0x1fu);

    matrix_accumulator_48 =
        uint32_t(combine_words(matrix_accumulator_48, 0u) << 0x5u >> 32) + matrix_fragment_47;

    matrix_fragment_46 = pack_half(matrix_accumulator_54, matrix_fragment_19);

    matrix_fragment_19 = uint32_t(int32_t(matrix_fragment_18) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_95), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_95 = result_word;
    }

    matrix_fragment_19 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x5u >> 32) + matrix_fragment_18;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_17), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    matrix_accumulator_48 = matrix_accumulator_48 & 0xffffffe0u;

    matrix_fragment_19 = matrix_fragment_19 & 0xffffffe0u;

    matrix_accumulator_49 = matrix_fragment_47 + (0u - matrix_accumulator_48);

    matrix_fragment_19 = matrix_fragment_18 + (0u - matrix_fragment_19);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_accumulator_49), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_49 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_19), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    matrix_fragment_95 = permute_signed(matrix_fragment_95, matrix_fragment_95, 0x5410u);

    matrix_fragment_17 = permute_signed(matrix_fragment_17, matrix_fragment_17, 0x5410u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_48 = permute_signed(matrix_fragment_18, matrix_fragment_18, 0x5410u);

    matrix_accumulator_49 = permute_signed(matrix_accumulator_49, matrix_accumulator_49, 0x5410u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                         select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_accumulator_48));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_143 + 0x1000u, matrix_accumulator_48, matrix_accumulator_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_3 = index_word_141 & 0x7ffffffcu;

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_143 + 0x1800u, matrix_accumulator_36, matrix_accumulator_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_20,
                             matrix_fragment_21);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_14,
                             matrix_fragment_15);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_12,
                             matrix_fragment_13);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_7 = 0u - matrix_accumulator_3 + index_word_0;

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(0u));

    index_word_142 = matrix_fragment_7 * 0x2u + index_word_142;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    index_word_141 = index_word_141 & 0xfffffffcu;

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = uint32_t(int32_t(index_word_142) >> 0x1u);

    matrix_fragment_9 = matrix_accumulator_113 + index_word_0 + (0u - index_word_141);

    matrix_fragment_13 = matrix_fragment_99 + index_word_0 + (0u - index_word_141);

    matrix_fragment_97 = matrix_fragment_97 + index_word_0 + (0u - index_word_141);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_fragment_13)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_fragment_17 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x30130ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_fragment_97)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_143 + 0x800u, matrix_fragment_32,
                                         matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_18 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_19 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x30130ull, memory_ranges,
        error_counts);

    matrix_fragment_52 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x30130ull, memory_ranges,
        error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2c130ull,
        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2c330ull,
        matrix_fragment_20, matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2d130ull,
        matrix_fragment_24, matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2d330ull,
        matrix_fragment_28, matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2e130ull,
        matrix_fragment_4, matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2e330ull,
        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2f130ull,
        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0x2f330ull,
        matrix_fragment_32, matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
        memory_ranges, error_counts);

    uniform_parameter_word_7 = 0u;

    uniform_parameter_word_6 = 0x400u;

    uniform_parameter_word_6 = (uniform_parameter_word_7 << 0x18u) + uniform_parameter_word_6;

    matrix_fragment_53 = (index_word_0 << 0x4u) + uniform_parameter_word_6;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53, matrix_accumulator_56, matrix_accumulator_57,
        matrix_accumulator_58, matrix_accumulator_59, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_48);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_49);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0x800u, matrix_fragment_44, matrix_fragment_45,
        matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_48 = unpack_fp8_bits(matrix_accumulator_48 >> 16);

    matrix_accumulator_49 = unpack_fp8_bits(matrix_accumulator_49 >> 16);

    uniform_index_word_4 = uniform_index_word_4 + 0x4u;

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_50);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_51);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_50 = unpack_fp8_bits(matrix_fragment_50 >> 16);

    matrix_fragment_51 = unpack_fp8_bits(matrix_fragment_51 >> 16);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_36);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_37);

    matrix_accumulator_36 = unpack_fp8_bits(matrix_accumulator_36 >> 16);

    matrix_accumulator_37 = unpack_fp8_bits(matrix_accumulator_37 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_36),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_37),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_56,
                             matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_50,
                             matrix_fragment_51);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0x200u, matrix_accumulator_48, matrix_accumulator_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_36,
                             matrix_accumulator_37);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0xa00u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_38);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_39);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_38 >> 16);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_39 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_38,
                             matrix_fragment_39);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0x400u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_48,
                             matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_accumulator_48,
                             matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_58,
                             matrix_accumulator_59);

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_36,
                             matrix_accumulator_37);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0xc00u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_48,
                             matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_48,
                             matrix_accumulator_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_38,
                             matrix_fragment_39);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0x600u, matrix_fragment_28, matrix_fragment_29,
        matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_54,
                             matrix_accumulator_55);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_56,
                             matrix_accumulator_57);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_53 + 0xe00u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_5 = index_word_0;

    matrix_fragment_5 =
        uint32_t(int64_t(int32_t(matrix_fragment_5)) * (1ll << 0x1cu) >> 32) + uniform_index_word_4;

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_2,
                             matrix_accumulator_3);

    condition_0 = int32_t(matrix_fragment_5) >= int32_t(index_word_128);

    condition_0 = int32_t(matrix_fragment_5) < int32_t(0u) || condition_0;

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_36,
                             matrix_accumulator_37);

    matrix_fragment_7 = matrix_fragment_5 + 0x2u;

    condition_1 = int32_t(matrix_fragment_7) >= int32_t(index_word_128);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x2u >> 32) + matrix_fragment_16;

    condition_1 = int32_t(matrix_fragment_7) < int32_t(0u) || condition_1;

    condition_2 = condition_0 | condition_3;

    condition_0 = condition_0 | condition_5;

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_fragment_9 = matrix_fragment_4 & 0xfffffffcu;

    condition_3 = condition_1 | condition_3;

    condition_1 = condition_1 | condition_5;

    multiply_accumulate_half(matrix_accumulator_54, matrix_accumulator_55, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_54,
                             matrix_accumulator_55);

    matrix_fragment_16 = matrix_fragment_16 + uniform_index_word_5 + (0u - matrix_fragment_9);

    if (!condition_2) {
        matrix_fragment_11 = index_word_128 * matrix_fragment_77 + matrix_fragment_5;
    }

    if (!condition_0) {
        matrix_fragment_8 = index_word_128 * matrix_fragment_81 + matrix_fragment_5;

        matrix_fragment_10 = index_word_128 * matrix_fragment_77 + matrix_fragment_5;
    }

    if (!condition_3) {
        matrix_fragment_9 = index_word_128 * matrix_fragment_81 + matrix_fragment_7;
    }

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_62,
                             matrix_accumulator_63);

    if (!condition_2) {
        matrix_fragment_6 = matrix_fragment_16 * 0x4u;
    }

    if (!condition_3) {
        matrix_fragment_4 = matrix_fragment_16 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_2,
                             matrix_accumulator_3);

    if (!condition_2) {
        matrix_accumulator_2 = matrix_fragment_16 * 0x4u;

        matrix_accumulator_3 = index_word_128 * matrix_fragment_81 + matrix_fragment_5;
    }

    if (!condition_3) {
        matrix_fragment_5 = index_word_128 * matrix_fragment_77 + matrix_fragment_7;
    }

    if (!condition_2) {
        matrix_accumulator_3 = matrix_fragment_76 * matrix_accumulator_3 + matrix_accumulator_2;
    }

    if (!condition_1) {
        matrix_fragment_81 = index_word_128 * matrix_fragment_81 + matrix_fragment_7;
    }

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_58,
                             matrix_accumulator_59);

    if (!condition_1) {
        matrix_fragment_77 = index_word_128 * matrix_fragment_77 + matrix_fragment_7;
    }

    if (!condition_0) {
        matrix_fragment_7 = matrix_fragment_80 * 0x4u;
    }

    if (!condition_3) {
        matrix_fragment_16 = matrix_fragment_16 * 0x4u;
    }

    if (!condition_1) {
        matrix_fragment_13 = matrix_fragment_80 * 0x4u;
    }

    matrix_accumulator_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_54),
                      select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_fragment_9 = matrix_fragment_76 * matrix_fragment_9 + matrix_fragment_4;
    }

    if (!condition_0) {
        matrix_accumulator_2 = matrix_fragment_76 * matrix_fragment_8 + matrix_fragment_7;
    }

    multiply_accumulate_half(matrix_accumulator_36, matrix_accumulator_37, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_36,
                             matrix_accumulator_37);

    if (!condition_2) {
        matrix_fragment_11 = matrix_fragment_76 * matrix_fragment_11 + matrix_fragment_6;
    }

    if (!condition_3) {
        matrix_fragment_5 = matrix_fragment_76 * matrix_fragment_5 + matrix_fragment_16;
    }

    if (!condition_1) {
        matrix_fragment_4 = matrix_fragment_76 * matrix_fragment_81 + matrix_fragment_13;
    }

    if (!condition_0) {
        matrix_fragment_15 = matrix_fragment_80 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_38,
                             matrix_fragment_39);

    if (!condition_0) {
        matrix_fragment_6 = matrix_fragment_76 * matrix_fragment_10 + matrix_fragment_15;
    }

    if (!condition_3) {
        matrix_fragment_25 = matrix_fragment_9 + index_word_0 + (0u - index_word_141);
    }

    if (!condition_1) {
        matrix_fragment_17 = matrix_fragment_80 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_27 = matrix_fragment_11 + index_word_0 + (0u - index_word_141);
    }

    if (!condition_3) {
        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_23 = matrix_accumulator_3 + index_word_0 + (0u - index_word_141);
    }

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_56,
                             matrix_accumulator_57);

    if (!condition_2) {
        matrix_fragment_43 = permute_signed(matrix_accumulator_54, matrix_accumulator_3, 0x5410u);
    }

    if (!condition_0) {
        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];

        matrix_fragment_41 = matrix_fragment_6 + index_word_0 + (0u - index_word_141);

        matrix_fragment_31 = matrix_accumulator_2 + index_word_0 + (0u - index_word_141);
    }

    if (!condition_3) {
        matrix_fragment_29 = matrix_fragment_5 + index_word_0 + (0u - index_word_141);
    }

    if (!condition_2) {
        matrix_accumulator_2 = parameters[parameter_output_low_2];
        matrix_accumulator_3 = parameters[parameter_output_high_3];
    }

    if (!condition_1) {
        matrix_fragment_35 = matrix_fragment_4 + index_word_0 + (0u - index_word_141);

        matrix_fragment_76 = matrix_fragment_76 * matrix_fragment_77 + matrix_fragment_17;
    }

    if (!condition_3) {
        matrix_fragment_4 = parameters[parameter_output_low_2];
        matrix_fragment_5 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_6 = parameters[parameter_output_low_2];
        matrix_fragment_7 = parameters[parameter_output_high_3];
    }

    if (!condition_1) {
        matrix_fragment_12 = parameters[parameter_output_low_2];
        matrix_fragment_13 = parameters[parameter_output_high_3];
    }

    if (!condition_0) {
        matrix_fragment_14 = parameters[parameter_output_low_2];
        matrix_fragment_15 = parameters[parameter_output_high_3];
    }

    if (!condition_1) {
        matrix_fragment_16 = parameters[parameter_output_low_2];
        matrix_fragment_17 = parameters[parameter_output_high_3];

        index_word_141 = matrix_fragment_76 + index_word_0 + (0u - index_word_141);
    }

    matrix_accumulator_55 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_55),
                      select_half_operand<2, false, false>(0u));

    index_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                 select_half_operand<2, false, false>(0u));

    matrix_accumulator_62 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_37),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_accumulator_55 = permute_signed(matrix_accumulator_55, index_word_0, 0x5410u);
    }

    if (!condition_2) {
        matrix_accumulator_37 = permute_signed(matrix_accumulator_62, matrix_fragment_19, 0x5410u);
    }

    if (!condition_3) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_31)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (!condition_3) {
        matrix_accumulator_63 = permute_signed(matrix_accumulator_63, matrix_fragment_18, 0x5410u);
    }

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    index_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                 select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                    matrix_fragment_43, memory_ranges, error_counts);
    }

    if (!condition_0) {
        matrix_fragment_21 = permute_signed(matrix_accumulator_58, matrix_fragment_21, 0x5410u);
    }

    if (!condition_1) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

        matrix_accumulator_59 = permute_signed(matrix_accumulator_59, matrix_fragment_20, 0x5410u);
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_accumulator_55, memory_ranges, error_counts);
    }

    if (!condition_0) {
        matrix_fragment_19 = permute_signed(matrix_fragment_32, matrix_fragment_19, 0x5410u);

        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_41)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_1) {
        matrix_fragment_33 = permute_signed(matrix_fragment_33, index_word_0, 0x5410u);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_accumulator_37, memory_ranges, error_counts);
    }

    if (!condition_1) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(index_word_141)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_accumulator_63, memory_ranges, error_counts);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_fragment_21, memory_ranges, error_counts);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_12, matrix_fragment_13),
                                    matrix_accumulator_59, memory_ranges, error_counts);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_14, matrix_fragment_15),
                                    matrix_fragment_19, memory_ranges, error_counts);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                    matrix_fragment_33, memory_ranges, error_counts);
    }

    __syncthreads();
    return;
}
extern "C" __global__ __maxnreg__(156) void phoenix_window_attention_channels_128_output_view(
    KernelArguments parameters) {
    run_window_attention_channels_128_output_view<false>(parameters.words, nullptr, nullptr);
}
