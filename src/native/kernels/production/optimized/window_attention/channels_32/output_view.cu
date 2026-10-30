#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: window_attention/channels_32/output_view.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (PHOENIX_STYLE_ARCH >= 900)
#define DLSS5_TILE_SHARED_BYTES 4
#include "../../../common/device_memory_scalar.cuh"
// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_output_low_2 = 2,
    parameter_output_high_3 = 3,
    parameter_trained_weights_low_4 = 4,
    parameter_trained_weights_high_5 = 5,
    parameter_tensor_height_6 = 6,
    parameter_tensor_width_7 = 7,
    parameter_optional_control_8 = 8,
    parameter_scalar_control_9 = 9,
    parameter_wait_optional_low_10 = 10,
    parameter_wait_optional_high_11 = 11,
    parameter_scalar_control_12 = 12,
    parameter_optional_control_13 = 13,
    parameter_optional_control_18 = 18,
    parameter_optional_control_19 = 19,
};

struct alignas(8) KernelArguments {
    uint32_t words[24];
};
static_assert(sizeof(KernelArguments) == 96, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_32_output_view(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
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
    uint32_t packed_half_13 = 0;
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
    uint32_t packed_half_29 = 0;
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
    uint32_t matrix_fragment_44 = 0;
    uint32_t matrix_fragment_45 = 0;
    uint32_t matrix_fragment_46 = 0;
    uint32_t matrix_fragment_47 = 0;
    uint32_t matrix_fragment_48 = 0;
    uint32_t matrix_fragment_49 = 0;
    uint32_t matrix_fragment_50 = 0;
    uint32_t matrix_fragment_51 = 0;
    uint32_t matrix_fragment_52 = 0;
    uint32_t matrix_fragment_53 = 0;
    uint32_t matrix_fragment_54 = 0;
    uint32_t matrix_fragment_55 = 0;
    uint32_t matrix_fragment_56 = 0;
    uint32_t matrix_fragment_57 = 0;
    uint32_t matrix_fragment_58 = 0;
    uint32_t matrix_fragment_59 = 0;
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
    uint32_t matrix_accumulator_72 = 0;
    uint32_t matrix_accumulator_73 = 0;
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_fragment_76 = 0;
    uint32_t matrix_fragment_77 = 0;
    uint32_t matrix_fragment_78 = 0;
    uint32_t matrix_fragment_79 = 0;
    uint32_t matrix_accumulator_80 = 0;
    uint32_t matrix_accumulator_81 = 0;
    uint32_t matrix_accumulator_82 = 0;
    uint32_t matrix_accumulator_83 = 0;
    uint32_t matrix_accumulator_84 = 0;
    uint32_t matrix_accumulator_85 = 0;
    uint32_t matrix_accumulator_86 = 0;
    uint32_t matrix_accumulator_87 = 0;
    uint32_t matrix_accumulator_88 = 0;
    uint32_t matrix_accumulator_89 = 0;
    uint32_t matrix_accumulator_90 = 0;
    uint32_t matrix_accumulator_91 = 0;
    uint32_t matrix_fragment_92 = 0;
    uint32_t matrix_fragment_93 = 0;
    uint32_t matrix_fragment_94 = 0;
    uint32_t matrix_fragment_95 = 0;
    uint32_t matrix_fragment_96 = 0;
    uint32_t matrix_fragment_97 = 0;
    uint32_t matrix_fragment_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t matrix_accumulator_100 = 0;
    uint32_t matrix_accumulator_101 = 0;
    uint32_t packed_half_102 = 0;
    uint32_t packed_half_103 = 0;
    uint32_t packed_half_104 = 0;
    uint32_t packed_half_105 = 0;
    uint32_t packed_half_106 = 0;
    uint32_t packed_half_107 = 0;
    uint32_t matrix_fragment_108 = 0;
    uint32_t matrix_fragment_109 = 0;
    uint32_t packed_half_110 = 0;
    uint32_t matrix_fragment_111 = 0;
    uint32_t packed_half_112 = 0;
    uint32_t packed_half_113 = 0;
    uint32_t packed_half_114 = 0;
    uint32_t packed_half_115 = 0;
    uint32_t packed_half_116 = 0;
    uint32_t packed_half_117 = 0;
    uint32_t matrix_accumulator_118 = 0;
    uint32_t matrix_accumulator_119 = 0;
    uint32_t packed_half_120 = 0;
    uint32_t packed_half_121 = 0;
    uint32_t packed_half_122 = 0;
    uint32_t matrix_accumulator_123 = 0;
    uint32_t matrix_accumulator_124 = 0;
    uint32_t matrix_accumulator_125 = 0;
    uint32_t matrix_accumulator_126 = 0;
    uint32_t matrix_accumulator_127 = 0;
    uint32_t packed_half_128 = 0;
    uint32_t packed_half_129 = 0;
    uint32_t packed_half_130 = 0;
    uint32_t packed_half_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t matrix_accumulator_133 = 0;
    uint32_t packed_half_134 = 0;
    uint32_t packed_half_135 = 0;
    uint32_t packed_half_136 = 0;
    uint32_t packed_half_137 = 0;
    uint32_t packed_half_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t packed_half_140 = 0;
    uint32_t packed_half_141 = 0;
    uint32_t packed_half_142 = 0;
    uint32_t packed_half_143 = 0;
    uint32_t packed_half_144 = 0;
    uint32_t packed_half_145 = 0;
    uint32_t packed_half_146 = 0;
    uint32_t packed_half_147 = 0;
    uint32_t packed_half_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t packed_half_150 = 0;
    uint32_t packed_half_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t address_word_154 = 0;
    uint32_t address_word_155 = 0;
    uint32_t address_word_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t address_word_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t address_word_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t float_bits_162 = 0;
    uint32_t float_bits_163 = 0;
    uint32_t address_word_164 = 0;
    uint32_t address_word_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_parameter_word_8 = 0;
    uint32_t uniform_block_index_9 = 0;
    uint32_t uniform_parameter_word_10 = 0;
    uint32_t uniform_parameter_word_11 = 0;
    uint32_t uniform_parameter_word_12 = 0;
    uint32_t uniform_parameter_word_13 = 0;
    uint32_t uniform_reused_word_14 = 0;
    uint32_t uniform_reused_word_15 = 0;
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

    packed_half_0 = blockIdx.y;

    uniform_parameter_word_7 = parameters[parameter_optional_control_13];

    matrix_fragment_8 = parameters[parameter_optional_control_8];
    matrix_fragment_9 = parameters[parameter_scalar_control_9];

    matrix_fragment_4 = threadIdx.x;

    uniform_index_word_6 = parameters[parameter_tensor_height_6];

    matrix_fragment_7 = threadIdx.y;

    uniform_parameter_word_8 = parameters[parameter_scalar_control_12];

    matrix_fragment_6 = threadIdx.z;

    uniform_index_word_5 = parameters[parameter_tensor_width_7];

    matrix_fragment_5 = blockIdx.x;

    uniform_reused_word_14 = 0u;
    uniform_reused_word_15 = 0u;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_7));

    uniform_index_word_6 = 0u - uniform_index_word_4 + uniform_index_word_6;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_8));

    uniform_condition_1 = uint32_t(uniform_index_word_6) != 0u;

    uniform_index_word_5 = 0u - uniform_index_word_4 + uniform_index_word_5;

    packed_half_0 = packed_half_0 * 0x8u + matrix_fragment_9;

    uniform_index_word_4 = 0u;

    uniform_condition_0 = uint32_t(uniform_index_word_5) != 0u;

    matrix_accumulator_2 = packed_half_0 + (0u - uniform_parameter_word_7);

    if (uniform_condition_1) {
        uniform_index_word_6 = uniform_index_word_6 + 0xffffffffu;
    }

    matrix_fragment_9 = matrix_accumulator_2 + 0x8u;

    matrix_accumulator_3 = matrix_accumulator_2 + 0x7u;

    if (uniform_condition_1) {
        uniform_parameter_word_7 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);
    }

    matrix_accumulator_2 = max(int32_t(0u), int32_t(matrix_accumulator_2));

    condition_0 = (matrix_fragment_4 | matrix_fragment_6 | matrix_fragment_7) != 0;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_5 + 0xffffffffu;
    }

    condition_1 = int32_t(matrix_fragment_9) > int32_t(0u);

    matrix_accumulator_3 = uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_1) {
        uniform_parameter_word_7 =
            uint32_t(combine_words(uniform_parameter_word_7, 0u) << 0x3u >> 32) +
            uniform_index_word_6;
    }

    matrix_fragment_4 = matrix_fragment_5 * 0x8u + matrix_fragment_8;

    matrix_accumulator_2 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

    if (uniform_condition_0) {
        uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_5) >> 0x1fu);
    }

    if (uniform_condition_1) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_3 = condition_1 ? matrix_accumulator_3 : 0u;

    matrix_fragment_5 = matrix_fragment_4 + (0u - uniform_parameter_word_8);

    if (uniform_condition_0) {
        uniform_index_word_6 =
            uint32_t(combine_words(uniform_index_word_6, 0u) << 0x3u >> 32) + uniform_index_word_5;
    }

    matrix_fragment_11 = min(int32_t(matrix_accumulator_3), int32_t(uniform_index_word_4));

    matrix_fragment_7 = matrix_fragment_5 + 0x8u;

    matrix_fragment_6 = matrix_fragment_5 + 0x7u;

    uniform_index_word_4 = 0u;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    condition_0 = int32_t(matrix_accumulator_2) >= int32_t(matrix_fragment_11) || condition_0;

    condition_1 = int32_t(matrix_fragment_7) > int32_t(0u);

    matrix_fragment_6 = uint32_t(combine_words(matrix_fragment_6, 0u) << 0x1du >> 32) + 0x1u;

    matrix_fragment_5 = max(int32_t(0u), int32_t(matrix_fragment_5));

    matrix_fragment_6 = condition_1 ? matrix_fragment_6 : 0u;
    if (!condition_0) {
        matrix_fragment_8 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_4));

        matrix_fragment_5 = uint32_t(uint32_t(matrix_fragment_5) >> 0x3u);

        matrix_fragment_6 = matrix_accumulator_2;
        do {
            condition_0 = int32_t(matrix_fragment_5) >= int32_t(matrix_fragment_8);
            if (!condition_0) {
                matrix_fragment_10 = matrix_fragment_6 * uniform_index_word_4;

                matrix_fragment_7 = matrix_fragment_5;
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_10];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_11];

                    matrix_fragment_9 = matrix_fragment_10 + matrix_fragment_7;

                    split_address_words(
                        matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

                    matrix_fragment_9 = pipeline_read<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    matrix_fragment_7 = matrix_fragment_7 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_7) != uint32_t(matrix_fragment_8);

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
            matrix_fragment_6 = matrix_fragment_6 + 0x1u;

            condition_0 = uint32_t(matrix_fragment_6) != uint32_t(matrix_fragment_11);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    matrix_fragment_8 = parameters[parameter_tensor_height_6];
    matrix_fragment_9 = parameters[parameter_tensor_width_7];

    matrix_accumulator_3 = uint32_t(int32_t(packed_half_0) >> 0x1fu);

    matrix_fragment_27 = threadIdx.x & 31u;

    condition_3 = false;

    uniform_parameter_word_10 = parameters[parameter_optional_control_8];
    uniform_parameter_word_11 = parameters[parameter_scalar_control_9];

    condition_5 = false;

    condition_4 = true;

    address_word_156 = parameters[parameter_trained_weights_low_4];
    packed_half_157 = parameters[parameter_trained_weights_high_5];

    __syncthreads();

    matrix_fragment_25 = 0u;

    uniform_index_word_6 = parameters[parameter_optional_control_18];
    uniform_parameter_word_7 = parameters[parameter_optional_control_19];

    address_word_154 = parameters[parameter_trained_weights_low_4];
    address_word_155 = parameters[parameter_trained_weights_high_5];

    uniform_parameter_word_12 = parameters[parameter_tensor_height_6];
    uniform_parameter_word_13 = parameters[parameter_tensor_width_7];

    matrix_accumulator_2 = matrix_fragment_8 & 0xfffffffcu;

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_8) >> 0x1fu);

    matrix_fragment_6 = uint32_t(int32_t(matrix_fragment_9) >> 0x1fu);

    condition_0 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + packed_half_0;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_8;

    matrix_accumulator_3 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    matrix_fragment_14 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + matrix_fragment_4;

    matrix_fragment_6 =
        uint32_t(combine_words(matrix_fragment_6, 0u) << 0x2u >> 32) + matrix_fragment_9;

    if (condition_0) {
        matrix_accumulator_2 = matrix_fragment_14 + 0x1u;
    }

    matrix_fragment_8 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    matrix_fragment_15 = uint32_t(int32_t(matrix_fragment_6) >> 0x2u);

    if (condition_0) {
        condition_1 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_5);

        condition_6 = int32_t(matrix_accumulator_2) >= int32_t(matrix_fragment_5);
    }

    matrix_fragment_9 = matrix_fragment_9 & 0xfffffffcu;

    condition_2 = int32_t(matrix_fragment_8) >= int32_t(matrix_fragment_15);

    if (condition_0) {
        condition_3 = int32_t(packed_half_0) < int32_t(0u - 0x3u) || condition_1;

        condition_5 = int32_t(packed_half_0) < int32_t(0u - 0x7u) || condition_6;
    }

    condition_1 = true;

    condition_2 = int32_t(matrix_fragment_4) > int32_t(0u - 0x4u) && !condition_2;

    condition_6 = uint32_t(matrix_fragment_9) == uint32_t(0x4u) || condition_3;

    if (condition_0) {
        condition_4 = !condition_3;

        condition_1 = !condition_5;
    }

    condition_4 = condition_4 & (condition_6 | condition_2);

    condition_6 = uint32_t(matrix_fragment_9) == uint32_t(0x4u) || condition_5;

    matrix_fragment_6 = uint32_t(condition_5) << 5;

    if (condition_0) {
        condition_5 = int32_t(matrix_accumulator_2) >= int32_t(matrix_fragment_5);
    }

    condition_1 = condition_1 & (condition_6 | condition_2);

    if (condition_0) {
        condition_2 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_5);
    }

    matrix_fragment_5 = condition_3 ? matrix_fragment_8 : 0u;

    condition_6 = false;

    if (condition_4) {
        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    condition_3 = uint32_t(matrix_fragment_9) != uint32_t(0x4u);

    if (condition_0) {
        condition_6 = int32_t(packed_half_0) < int32_t(0u - 0x3u) || condition_2;
    }

    condition_2 = false;

    if (condition_0) {
        condition_2 = int32_t(packed_half_0) < int32_t(0u - 0x7u) || condition_5;
    }

    packed_half_0 = 0u;

    if (condition_0) {
        packed_half_0 = matrix_fragment_14 * matrix_fragment_15;
    }

    matrix_fragment_5 = !condition_3 ? matrix_fragment_5 : matrix_fragment_8;

    if (!condition_4) {
        matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_4) {
        packed_half_0 = matrix_fragment_5 + packed_half_0;
    }

    if (!condition_4) {
        matrix_fragment_23 = permute_signed(matrix_fragment_23, matrix_fragment_23, 0x5410u);
    }

    if (condition_4) {
        matrix_fragment_5 = packed_half_0 * 0x80u;
    }

    if (!condition_4) {
        matrix_fragment_20 = matrix_fragment_23;

        matrix_fragment_21 = matrix_fragment_23;

        matrix_fragment_22 = matrix_fragment_23;
    }

    packed_half_0 = matrix_fragment_8 + 0x1u;

    if (condition_4) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    condition_5 = true;

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }

    condition_4 = int32_t(packed_half_0) >= int32_t(matrix_fragment_15);

    if (condition_0) {
        condition_5 = !condition_6;
    }

    split_address_words(matrix_fragment_44, matrix_fragment_45,
                        int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_156, packed_half_157));

    matrix_fragment_7 = condition_6 ? packed_half_0 : 0u;

    condition_6 = uint32_t(matrix_fragment_9) == uint32_t(0x4u) || condition_6;

    condition_4 = int32_t(matrix_fragment_4) > int32_t(0u - 0x8u) && !condition_4;

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_44, matrix_fragment_45),
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    matrix_accumulator_2 = 0u;

    condition_6 = condition_5 & (condition_6 | condition_4);

    if (condition_0) {
        matrix_accumulator_2 = matrix_fragment_14 * matrix_fragment_15;
    }

    matrix_fragment_7 = !condition_3 ? matrix_fragment_7 : packed_half_0;

    condition_5 = int32_t(matrix_fragment_6) != int32_t(0u);

    matrix_fragment_10 = matrix_fragment_27;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1000ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1200ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    if (condition_6) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];

        matrix_accumulator_2 = matrix_fragment_7 + matrix_accumulator_2;

        matrix_accumulator_3 = matrix_accumulator_2 * 0x80u;

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x200ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    matrix_fragment_10 = matrix_fragment_10 & 0xfffffffcu;

    if (!condition_6) {
        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    address_word_165 = 0u - matrix_fragment_10 + matrix_fragment_27;

    if (!condition_6) {
        matrix_fragment_19 = permute_signed(matrix_fragment_19, matrix_fragment_19, 0x5410u);
    }

    if (condition_6) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    split_address_words(address_word_164, address_word_165,
                        int64_t(int32_t(address_word_165)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_156, packed_half_157));

    if (!condition_6) {
        matrix_fragment_16 = matrix_fragment_19;

        matrix_fragment_17 = matrix_fragment_19;

        matrix_fragment_18 = matrix_fragment_19;
    }

    matrix_fragment_48 = load_aligned_word<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x2010ull, memory_ranges, error_counts);

    address_word_161 = matrix_fragment_27 + 0x4u + (0u - matrix_fragment_10);

    if (condition_6) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    address_word_159 = matrix_fragment_27 + 0x8u + (0u - matrix_fragment_10);

    split_address_words(address_word_160, address_word_161,
                        uint64_t(uint32_t(address_word_161)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_156, packed_half_157));

    split_address_words(address_word_158, address_word_159,
                        uint64_t(uint32_t(address_word_159)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_156, packed_half_157));

    matrix_fragment_58 = load_aligned_word<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x2010ull, memory_ranges, error_counts);

    matrix_fragment_57 = load_aligned_word<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x2010ull, memory_ranges, error_counts);

    matrix_accumulator_3 = matrix_fragment_27 + 0xcu + (0u - matrix_fragment_10);

    split_address_words(address_word_156, packed_half_157,
                        uint64_t(uint32_t(matrix_accumulator_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_156, packed_half_157));

    if (condition_1) {
        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    matrix_fragment_56 = load_aligned_word<CheckBounds>(
        combine_words(address_word_156, packed_half_157) + 0x2010ull, memory_ranges, error_counts);

    if (!condition_3) {
        matrix_fragment_8 = condition_5 ? matrix_fragment_8 : 0u;
    }

    condition_6 = uint32_t(matrix_fragment_9) == uint32_t(0x4u) || condition_2;

    condition_5 = true;

    if (condition_0) {
        condition_5 = !condition_2;
    }

    matrix_fragment_9 = 0u;

    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_14 * matrix_fragment_15 + matrix_fragment_15;
    }

    condition_5 = condition_5 & (condition_6 | condition_4);

    if (condition_1) {
        matrix_fragment_9 = matrix_fragment_8 + matrix_fragment_9;
    }

    if (!condition_1) {
        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_1) {
        matrix_fragment_9 = matrix_fragment_9 * 0x80u;
    }

    if (!condition_1) {
        matrix_fragment_11 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5410u);
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (condition_5) {
        matrix_fragment_12 = parameters[parameter_input_low_0];
        packed_half_13 = parameters[parameter_input_high_1];
    }

    if (!condition_1) {
        matrix_fragment_8 = matrix_fragment_11;

        matrix_fragment_9 = matrix_fragment_11;

        matrix_fragment_10 = matrix_fragment_11;
    }

    if (condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (!condition_3) {
        packed_half_0 = condition_2 ? packed_half_0 : 0u;
    }

    if (condition_0) {
        matrix_fragment_25 = matrix_fragment_14 * matrix_fragment_15 + matrix_fragment_15;
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    if (condition_5) {
        matrix_fragment_25 = packed_half_0 + matrix_fragment_25;
    }

    if (!condition_5) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_5) {
        matrix_fragment_25 = matrix_fragment_25 * 0x80u;
    }

    if (!condition_5) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_5) {
        split_address_words(matrix_fragment_12, packed_half_13,
                            int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, packed_half_13));
    }

    if (!condition_5) {
        matrix_fragment_14 = matrix_fragment_15;
    }

    if (condition_5) {
        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_12, packed_half_13));
    }

    if (!condition_5) {
        matrix_fragment_12 = matrix_fragment_15;

        packed_half_13 = matrix_fragment_15;
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25), matrix_fragment_12,
            packed_half_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);
    }

    packed_half_0 = 0x2b28u;

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_28, packed_half_29, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_25),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_29), 0x44004400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    packed_half_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_26),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_29),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_29),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    packed_half_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(packed_half_29));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_20);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    packed_half_29 = unpack_fp8_bits(matrix_fragment_21);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    packed_half_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                                         select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                             packed_half_29);

    packed_half_29 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    packed_half_29 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_22);

    packed_half_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                                         select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_30 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x44004400u);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_28,
                             packed_half_29);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    packed_half_29 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    packed_half_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                                         select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_28,
                             packed_half_29);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_30),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_31),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    packed_half_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_31),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_30 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_29), 0xc400c400u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_accumulator_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

    matrix_accumulator_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x400ull, matrix_fragment_28,
        packed_half_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_64),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3b283b28u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3b283b28u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x600ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_18);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_19);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_accumulator_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_fragment_41 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x44004400u);

    matrix_fragment_40 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x44004400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_40),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3b283b28u);

    matrix_accumulator_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_accumulator_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_fragment_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0xc400c400u);

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_76),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_82),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<2, false, false>(matrix_fragment_77), 0x3b283b28u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3b283b28u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_49 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_50 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_82,
                             matrix_accumulator_83);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1400ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    matrix_fragment_92 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_93 = unpack_fp8_bits(packed_half_13);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_4),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_6),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    matrix_fragment_51 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x44004400u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_6),
        select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

    matrix_fragment_6 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x44004400u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    matrix_fragment_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_6), 0x3b283b28u);

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x44004400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1600ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x800ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_86),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3b283b28u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_87 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_accumulator_87));

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_15);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_5 = unpack_fp8_bits(packed_half_13 >> 16);

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0xa00ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x44004400u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0xc400c400u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_99),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_96),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3b283b28u);

    matrix_fragment_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x44004400u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<2, false, false>(matrix_fragment_98), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0xc400c400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0xc400c400u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<2, false, false>(matrix_fragment_96), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_98),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_102), 0xc400c400u);

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3b283b28u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0xc400c400u);

    matrix_fragment_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_106),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0xc400c400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0xc400c400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<2, false, false>(packed_half_107), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_28, packed_half_29, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_65),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                          select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_94),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<2, false, false>(matrix_fragment_95), 0x3b283b28u);

    matrix_accumulator_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0xc400c400u);

    matrix_accumulator_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_102 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0xc400c400u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0xc400c400u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_94),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_25),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_102),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3b283b28u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    packed_half_102 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(packed_half_102));

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_29), 0x44004400u);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(packed_half_103));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(matrix_fragment_95));

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0xc400c400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_32),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_33),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_33),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_34),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<2, false, false>(matrix_fragment_35), 0x3b283b28u);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0xc400c400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(matrix_fragment_111), 0x3b283b28u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_29 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                                        0x44004400u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_111));

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_30 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    packed_half_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_29), 0xc400c400u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                          select_half_operand<2, false, false>(packed_half_113));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1800ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_29),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_57),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_29),
        select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    packed_half_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_109),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_97),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<2, false, false>(matrix_fragment_38), 0x3b283b28u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1a00ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_108),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_111),
                                          select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_29));

    matrix_accumulator_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                      select_half_operand<2, false, false>(matrix_fragment_31));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0xe00ull, matrix_fragment_28,
        packed_half_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_accumulator_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                          select_half_operand<2, false, false>(0u));

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                      select_half_operand<2, false, false>(matrix_accumulator_61));

    packed_half_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                    select_half_operand<2, false, false>(0u));

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0xc400c400u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    matrix_accumulator_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                          select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                        select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_66,
                             matrix_accumulator_67);

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_106),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_accumulator_66 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                      select_half_operand<2, false, false>(matrix_accumulator_67));

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    matrix_accumulator_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                          select_half_operand<2, false, false>(packed_half_105));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0xc00ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_accumulator_65 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                      select_half_operand<2, false, false>(matrix_fragment_108));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                          select_half_operand<2, false, false>(packed_half_110));

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                          select_half_operand<2, false, false>(packed_half_106));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(packed_half_104));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_2,
                             matrix_accumulator_3);

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(packed_half_106));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_68 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(packed_half_103));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x44004400u);

    matrix_fragment_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_92,
                             matrix_fragment_93);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_90,
                             matrix_accumulator_91);

    matrix_fragment_69 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x44004400u);

    matrix_accumulator_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_71),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_58 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x44004400u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<2, false, false>(packed_half_102), 0x3b283b28u);

    matrix_fragment_59 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x44004400u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x44004400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3b283b28u);

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    packed_half_107 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0xc400c400u);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    packed_half_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0xc400c400u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0xc400c400u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3b283b28u);

    matrix_fragment_68 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(matrix_fragment_108), 0x3b283b28u);

    matrix_fragment_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108), 0xc400c400u);

    packed_half_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(packed_half_102));

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_106),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

    matrix_fragment_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3b283b28u);

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_109),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(packed_half_105));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_4),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_111),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    packed_half_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3b283b28u);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_110));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    packed_half_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_fragment_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_108));

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109), 0xc400c400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_6), 0x3b283b28u);

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_108),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_40 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x44004400u);

    matrix_fragment_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(matrix_fragment_111), 0x3b283b28u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_108),
        select_half_operand<2, false, false>(matrix_fragment_109), 0x3b283b28u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x44004400u);

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_111));

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x44004400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_56 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_40),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_108 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x44004400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_108), 0x3b283b28u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_111),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_108));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_57),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x44004400u);

    matrix_fragment_6 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x44004400u);

    matrix_fragment_4 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x44004400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1c00ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

    matrix_accumulator_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

    matrix_accumulator_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x1e00ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_71),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_109 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_109),
                                        select_half_operand<2, false, false>(0u));

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3b283b28u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<2, false, false>(packed_half_102), 0x3b283b28u);

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(0u));

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    matrix_fragment_108 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_108),
                                        select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

    packed_half_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(packed_half_102));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    packed_half_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(matrix_fragment_108));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(0u));

    packed_half_102 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_102),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_accumulator_62 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                      select_half_operand<2, false, false>(matrix_accumulator_63));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_accumulator_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                          select_half_operand<2, false, false>(packed_half_102));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(packed_half_102, packed_half_103, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_82,
                             matrix_accumulator_83);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x2260ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_2,
                             matrix_accumulator_3);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x2060ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    matrix_accumulator_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                      select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                      select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x2460ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_60,
                             matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_accumulator_86 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    matrix_fragment_42 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    matrix_accumulator_87 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    matrix_fragment_17 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x44004400u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_16),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_41),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<2, false, false>(matrix_fragment_17), 0x3b283b28u);

    matrix_fragment_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_40),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_43),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_87),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<2, false, false>(matrix_fragment_16), 0x3b283b28u);

    matrix_fragment_40 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x44004400u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3b283b28u);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87),
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3b283b28u);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, packed_half_29, 0u, 0u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_fragment_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    matrix_fragment_16 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x44004400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x2660ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    matrix_fragment_28 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x44004400u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    packed_half_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_16),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_29),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_29),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    packed_half_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_accumulator_88));

    matrix_accumulator_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_29), 0xc400c400u);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_89 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0xc400c400u);

    packed_half_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_accumulator_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_89),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3b283b28u);

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0xc400c400u);

    matrix_accumulator_88 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x44004400u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89),
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3b283b28u);

    matrix_accumulator_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    matrix_accumulator_89 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0xc400c400u);

    matrix_accumulator_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    matrix_fragment_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

    matrix_fragment_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_83),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0xc400c400u);

    matrix_accumulator_91 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0xc400c400u);

    matrix_fragment_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<2, false, false>(matrix_fragment_10), 0x3b283b28u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_11),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3b283b28u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_11),
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3b283b28u);

    matrix_fragment_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0xc400c400u);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0xc400c400u);

    matrix_accumulator_86 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0xc400c400u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_82),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_91),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_86),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_90 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0xc400c400u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_14),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

    matrix_accumulator_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_82),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_90),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_86 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0xc400c400u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_accumulator_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90),
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3b283b28u);

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(packed_half_13), 0x3b283b28u);

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0xc400c400u);

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_86),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_fragment_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x44004400u);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3b283b28u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_22 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x44004400u);

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                         select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_106),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_23 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

    matrix_fragment_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_14),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_21),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<2, false, false>(packed_half_107), 0x3b283b28u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3b283b28u);

    matrix_fragment_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<2, false, false>(matrix_fragment_19), 0x3b283b28u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_21),
        select_half_operand<2, false, false>(matrix_fragment_22), 0x3b283b28u);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_20),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                            select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_23),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                                      select_half_operand<2, false, false>(0u));

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_23),
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3b283b28u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_29),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                   select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_108),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_29), 0xc400c400u);

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_108),
        select_half_operand<2, false, false>(matrix_fragment_109), 0x3b283b28u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_86),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_29),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_8),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3b283b28u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_29),
        select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x2860ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_46,
                             matrix_fragment_47);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x2a60ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, packed_half_102,
                             packed_half_103);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_12,
                             packed_half_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_28, packed_half_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_34,
                             matrix_fragment_35);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_110, matrix_fragment_111, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(packed_half_0));

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                   select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                   select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_29),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                         select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    packed_half_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                      select_half_operand<2, false, false>(packed_half_110),
                                      select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                                      select_half_operand<2, false, false>(matrix_accumulator_66),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                        select_half_operand<2, false, false>(packed_half_0));

    packed_half_13 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4c60ull, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_56,
                             matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(packed_half_130, packed_half_131, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_58,
                             matrix_fragment_59, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                                      select_half_operand<2, false, false>(matrix_accumulator_67),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                                      select_half_operand<2, false, false>(matrix_fragment_111),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                      select_half_operand<2, false, false>(packed_half_128),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    multiply_accumulate_half(packed_half_148, packed_half_149, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_92,
                             matrix_fragment_93, 0u, 0u);

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                      select_half_operand<2, false, false>(packed_half_130),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    multiply_accumulate_half(packed_half_144, packed_half_145, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_92,
                             matrix_fragment_93, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_92, matrix_fragment_93, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_92, matrix_fragment_93, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                      select_half_operand<2, false, false>(packed_half_129),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                      select_half_operand<2, false, false>(packed_half_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_0 = result_word;
    }

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    multiply_accumulate_half(packed_half_150, packed_half_151, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_42,
                             matrix_fragment_43, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_12 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_94, matrix_fragment_95, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    multiply_accumulate_half(packed_half_140, packed_half_141, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_40,
                             matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(packed_half_146, packed_half_147, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_42,
                             matrix_fragment_43, 0u, 0u);

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                   select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(packed_half_142, packed_half_143, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_94,
                             matrix_fragment_95, 0u, 0u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_fragment_15 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 = permute_signed(matrix_accumulator_3, matrix_accumulator_3, 0x5432u);

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                      select_half_operand<2, false, false>(packed_half_148),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_12));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(packed_half_138, matrix_accumulator_139, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                             select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                             select_half_operand<2, false, false>(packed_half_146));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_94, matrix_fragment_95, 0u, 0u);

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                      select_half_operand<2, false, false>(packed_half_149),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                                      select_half_operand<2, false, false>(matrix_accumulator_63),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(packed_half_134, packed_half_135, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                             select_half_operand<2, false, false>(packed_half_150));

    matrix_fragment_40 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                                      select_half_operand<2, false, false>(matrix_accumulator_62),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                             select_half_operand<2, false, false>(packed_half_140));

    matrix_fragment_17 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                      select_half_operand<2, false, false>(packed_half_142),
                                      select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                      select_half_operand<2, false, false>(packed_half_144),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_17 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                             select_half_operand<2, false, false>(packed_half_141));

    matrix_fragment_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                             select_half_operand<2, false, false>(packed_half_147));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_97));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    packed_half_104 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x4100410u);

    matrix_fragment_41 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                      select_half_operand<2, false, false>(packed_half_143),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_40 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                                             select_half_operand<2, false, false>(packed_half_138));

    multiply_accumulate_half(packed_half_136, packed_half_137, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_17 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_17));

    packed_half_0 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x4100410u);

    matrix_fragment_40 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_15 = half_to_float_bits((matrix_accumulator_3 & 65535u) * 65537u);

    matrix_fragment_16 = half_to_float_bits((matrix_accumulator_3 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                     select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_40 = half_to_float_bits((packed_half_0 & 65535u) * 65537u);

    matrix_fragment_42 = half_to_float_bits((packed_half_0 >> 16) * 65537u);

    packed_half_0 = threadIdx.x & 31u;

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_94, matrix_fragment_95, 0u, 0u);

    matrix_fragment_109 = rsqrt_bits(matrix_fragment_15);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                     select_half_operand<2, false, false>(packed_half_104));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_101 = result_word;
    }

    matrix_fragment_16 = rsqrt_bits(matrix_fragment_16);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_104), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_100 = result_word;
    }

    matrix_fragment_40 = rsqrt_bits(matrix_fragment_40);

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_43 = rsqrt_bits(matrix_fragment_42);

    matrix_fragment_109 = pack_half(matrix_fragment_16, matrix_fragment_109);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_97 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                             select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                             select_half_operand<2, false, false>(packed_half_137));

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    packed_half_112 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_97 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_42 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                      select_half_operand<2, false, false>(matrix_fragment_95),
                                      select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_98 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(matrix_fragment_97));

    split_address_words(address_word_154, address_word_155,
                        int64_t(int32_t(packed_half_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_154, address_word_155));

    matrix_fragment_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                             select_half_operand<2, false, false>(packed_half_134));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_107 = result_word;
    }

    matrix_fragment_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                             select_half_operand<2, false, false>(packed_half_136));

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                     select_half_operand<2, false, false>(matrix_fragment_41));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_98), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_101 = result_word;
    }

    matrix_fragment_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_16 = pack_half(matrix_fragment_43, matrix_fragment_40);

    matrix_fragment_99 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                      select_half_operand<2, false, false>(matrix_fragment_94),
                                      select_half_operand<2, false, false>(matrix_fragment_99));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_114 = result_word;
    }

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x2c60ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_104), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_113 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_105), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_102 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_112), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_103 = result_word;
    }

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_fragment_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    matrix_fragment_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_109));

    matrix_fragment_36 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5432u);

    packed_half_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_68,
                             matrix_fragment_69, 0u, 0u);

    load_four_words_cached<CheckBounds>(combine_words(address_word_154, address_word_155) +
                                            0x3060ull,
                                        packed_half_116, packed_half_117, matrix_accumulator_118,
                                        matrix_accumulator_119, memory_ranges, error_counts);

    matrix_fragment_36 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x4100410u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_98 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(packed_half_114));

    matrix_accumulator_64 = half_to_float_bits((matrix_fragment_37 & 65535u) * 65537u);

    packed_half_114 = half_to_float_bits((matrix_fragment_37 >> 16) * 65537u);

    matrix_fragment_37 = permute_signed(matrix_fragment_12, matrix_fragment_12, 0x5432u);

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                     select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_65 = permute_signed(matrix_fragment_98, matrix_fragment_98, 0x5432u);

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                     select_half_operand<2, false, false>(packed_half_102));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(packed_half_103));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x2e60ull, matrix_accumulator_100,
        matrix_accumulator_101, packed_half_102, packed_half_103, memory_ranges, error_counts);

    matrix_accumulator_66 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_37 = permute_signed(packed_half_104, packed_half_104, 0x5432u);

    matrix_fragment_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_105), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_106 = result_word;
    }

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    multiply_accumulate_half(packed_half_122, matrix_accumulator_123, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_70, matrix_fragment_71, 0u, 0u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_112), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_65 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66), 0x4100410u);

    packed_half_106 = add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                     select_half_operand<2, false, false>(packed_half_106));

    matrix_accumulator_64 = rsqrt_bits(matrix_accumulator_64);

    matrix_accumulator_67 = half_to_float_bits((packed_half_107 & 65535u) * 65537u);

    packed_half_104 = half_to_float_bits((packed_half_107 >> 16) * 65537u);

    matrix_accumulator_66 = half_to_float_bits((matrix_fragment_6 & 65535u) * 65537u);

    packed_half_107 = half_to_float_bits((matrix_fragment_6 >> 16) * 65537u);

    packed_half_105 = rsqrt_bits(packed_half_114);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = permute_signed(packed_half_106, packed_half_106, 0x5432u);

    packed_half_106 = add_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                     select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_67 = rsqrt_bits(matrix_accumulator_67);

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(matrix_accumulator_65));

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x4100410u);

    packed_half_104 = rsqrt_bits(packed_half_104);

    matrix_accumulator_65 = half_to_float_bits((matrix_fragment_6 & 65535u) * 65537u);

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_152 = half_to_float_bits((matrix_fragment_6 >> 16) * 65537u);

    packed_half_105 = pack_half(packed_half_105, matrix_accumulator_64);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3260ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    packed_half_153 = half_to_float_bits((packed_half_106 & 65535u) * 65537u);

    matrix_accumulator_64 = half_to_float_bits((packed_half_106 >> 16) * 65537u);

    matrix_accumulator_66 = rsqrt_bits(matrix_accumulator_66);

    packed_half_107 = rsqrt_bits(packed_half_107);

    matrix_accumulator_67 = pack_half(packed_half_104, matrix_accumulator_67);

    matrix_accumulator_65 = rsqrt_bits(matrix_accumulator_65);

    packed_half_152 = rsqrt_bits(packed_half_152);

    packed_half_153 = rsqrt_bits(packed_half_153);

    matrix_accumulator_64 = rsqrt_bits(matrix_accumulator_64);

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_96 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                             select_half_operand<2, false, false>(packed_half_105));

    matrix_accumulator_66 = pack_half(packed_half_107, matrix_accumulator_66);

    packed_half_148 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                      select_half_operand<2, false, false>(0u));

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                          select_half_operand<2, false, false>(packed_half_105));

    packed_half_152 = pack_half(packed_half_152, matrix_accumulator_65);

    packed_half_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    packed_half_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                            select_half_operand<2, false, false>(matrix_accumulator_67));

    packed_half_13 = pack_half(0u, packed_half_13);

    packed_half_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_accumulator_67));

    packed_half_151 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                            select_half_operand<2, false, false>(matrix_accumulator_67));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3460ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    packed_half_144 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                            select_half_operand<2, false, false>(matrix_accumulator_66));

    packed_half_153 = pack_half(matrix_accumulator_64, packed_half_153);

    packed_half_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                            select_half_operand<2, false, false>(matrix_accumulator_66));

    packed_half_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                            select_half_operand<2, false, false>(matrix_accumulator_66));

    packed_half_146 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                            select_half_operand<2, false, false>(matrix_accumulator_66));

    packed_half_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                    select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3660ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109),
                            select_half_operand<0, false, false>(packed_half_13));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3860ull, packed_half_104,
        packed_half_105, packed_half_106, packed_half_107, memory_ranges, error_counts);

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_111 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_111),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(packed_half_114));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<0, false, false>(packed_half_13));

    packed_half_114 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_109),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(packed_half_13));

    packed_half_115 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                        select_half_operand<2, false, false>(matrix_fragment_111));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                    select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_17 = permute_signed(packed_half_112, packed_half_112, 0x5432u);

    matrix_fragment_108 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_108),
                                        select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3a60ull, packed_half_112,
        packed_half_113, packed_half_114, packed_half_115, memory_ranges, error_counts);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x4100410u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(0u));

    float_bits_162 = half_to_float_bits((matrix_fragment_17 & 65535u) * 65537u);

    float_bits_163 = half_to_float_bits((matrix_fragment_17 >> 16) * 65537u);

    matrix_fragment_111 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    float_bits_162 = rsqrt_bits(float_bits_162);

    matrix_fragment_15 = rsqrt_bits(float_bits_163);

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_152));

    packed_half_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(packed_half_152));

    packed_half_141 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                          select_half_operand<2, false, false>(packed_half_152));

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                          select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_15 = pack_half(matrix_fragment_15, float_bits_162);

    packed_half_147 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                          select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(packed_half_153));

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                                          select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                       select_half_operand<2, false, false>(0u));

    packed_half_152 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(matrix_fragment_92));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_96, matrix_fragment_97, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                       select_half_operand<2, false, false>(0u));

    packed_half_150 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                          select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                          select_half_operand<2, false, false>(matrix_fragment_15));

    packed_half_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = permute_signed(matrix_fragment_14, matrix_fragment_14, 0x5432u);

    matrix_accumulator_139 = permute_signed(matrix_fragment_12, matrix_fragment_12, 0x5432u);

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                       select_half_operand<2, false, false>(0u));

    packed_half_138 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                     select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x4100410u);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                                       select_half_operand<2, false, false>(packed_half_152));

    packed_half_143 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_138), 0x4100410u);

    packed_half_138 = permute_signed(matrix_accumulator_3, matrix_accumulator_3, 0x5432u);

    packed_half_140 = half_to_float_bits((matrix_fragment_35 & 65535u) * 65537u);

    packed_half_141 = half_to_float_bits((matrix_fragment_35 >> 16) * 65537u);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(0u));

    packed_half_142 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                     select_half_operand<2, false, false>(packed_half_138));

    multiply_accumulate_half(packed_half_138, matrix_accumulator_139, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_92, matrix_fragment_93, packed_half_102,
                             packed_half_103);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_3 = half_to_float_bits((packed_half_143 & 65535u) * 65537u);

    packed_half_142 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_142), 0x4100410u);

    matrix_fragment_30 = half_to_float_bits((packed_half_143 >> 16) * 65537u);

    packed_half_103 = permute_signed(matrix_accumulator_2, matrix_accumulator_2, 0x5432u);

    packed_half_143 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                    select_half_operand<2, false, false>(0u));

    packed_half_103 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                     select_half_operand<2, false, false>(packed_half_103));

    packed_half_102 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(packed_half_143));

    matrix_fragment_12 = rsqrt_bits(packed_half_140);

    matrix_accumulator_2 = half_to_float_bits((packed_half_142 & 65535u) * 65537u);

    packed_half_143 = half_to_float_bits((packed_half_142 >> 16) * 65537u);

    packed_half_142 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0x4100410u);

    matrix_accumulator_3 = rsqrt_bits(matrix_accumulator_3);

    packed_half_140 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = rsqrt_bits(matrix_fragment_30);

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                    select_half_operand<2, false, false>(packed_half_140));

    packed_half_140 = half_to_float_bits((packed_half_142 & 65535u) * 65537u);

    packed_half_142 = half_to_float_bits((packed_half_142 >> 16) * 65537u);

    packed_half_141 = rsqrt_bits(packed_half_141);

    matrix_accumulator_2 = rsqrt_bits(matrix_accumulator_2);

    packed_half_143 = rsqrt_bits(packed_half_143);

    matrix_accumulator_3 = pack_half(matrix_fragment_30, matrix_accumulator_3);

    packed_half_140 = rsqrt_bits(packed_half_140);

    packed_half_145 = rsqrt_bits(packed_half_142);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    packed_half_147 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                    select_half_operand<2, false, false>(0u));

    packed_half_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_12 = pack_half(packed_half_141, matrix_fragment_12);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(packed_half_147));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                         select_half_operand<2, false, false>(0u));

    packed_half_136 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_2 = pack_half(packed_half_143, matrix_accumulator_2);

    packed_half_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                          select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    packed_half_140 = pack_half(packed_half_145, packed_half_140);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                          select_half_operand<2, false, false>(packed_half_140));

    packed_half_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    multiply_accumulate_half(packed_half_116, packed_half_117, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_34, matrix_fragment_35, packed_half_116,
                             packed_half_117);

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(matrix_fragment_12));

    packed_half_130 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_130),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(packed_half_140));

    packed_half_131 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_131),
                                          select_half_operand<0, false, false>(packed_half_13));

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             packed_half_102, packed_half_103, matrix_accumulator_118,
                             matrix_accumulator_119);

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                       select_half_operand<2, false, false>(0u));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_140));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_137),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_12 = 0x29c0u;

    multiply_accumulate_half(matrix_fragment_108, matrix_fragment_109, matrix_fragment_108,
                             matrix_fragment_109, packed_half_110, matrix_fragment_111,
                             matrix_fragment_94, matrix_fragment_95, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<0, false, false>(packed_half_13));

    packed_half_13 = permute_signed(packed_half_13, matrix_fragment_12, 0x5410u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(0u));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    packed_half_138 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_138),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x3c203c20u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_116),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    matrix_accumulator_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_96, matrix_fragment_97, matrix_accumulator_60,
                             matrix_accumulator_61);

    packed_half_138 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_138), 0x3c203c20u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x3e473e47u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    packed_half_138 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_138), 0x3e473e47u);

    packed_half_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_116), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_accumulator_118 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x3c203c20u);

    packed_half_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_116), 0x3e473e47u);

    matrix_accumulator_118 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118), 0x3e473e47u);

    packed_half_153 = (matrix_fragment_16 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_64,
                             matrix_accumulator_65);

    packed_half_152 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    packed_half_151 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    packed_half_149 = (packed_half_138 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_66,
                             matrix_accumulator_67);

    packed_half_143 = (packed_half_116 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                        select_half_operand<2, false, false>(packed_half_152));

    packed_half_141 = (matrix_accumulator_118 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                        select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                        select_half_operand<2, false, false>(packed_half_141));

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_34,
                             matrix_fragment_35, packed_half_104, packed_half_105);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_2),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_108),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_17),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_102, packed_half_103,
                             packed_half_106, packed_half_107);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_15),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_139 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_117),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108), 0x3c203c20u);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_30,
                             matrix_fragment_31, packed_half_112, packed_half_113);

    matrix_accumulator_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x3c203c20u);

    matrix_accumulator_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_94, matrix_fragment_95, packed_half_114,
                             packed_half_115);

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3c203c20u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_accumulator_139 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3c203c20u);

    packed_half_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_117), 0x3c203c20u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    matrix_fragment_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_108), 0x3e473e47u);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_accumulator_62 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3c203c20u);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x3e473e47u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3e473e47u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_139 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3e473e47u);

    matrix_accumulator_119 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3c203c20u);

    packed_half_117 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_117), 0x3e473e47u);

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_accumulator_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3e473e47u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    packed_half_145 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    packed_half_144 = (matrix_fragment_108 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_119 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3e473e47u);

    packed_half_129 = (matrix_fragment_17 << 0x5u) + 0x7ff88000u;

    packed_half_128 = (matrix_fragment_15 << 0x5u) + 0x7ff88000u;

    packed_half_150 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    packed_half_148 = (matrix_accumulator_139 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(packed_half_144));

    packed_half_142 = (packed_half_117 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    packed_half_138 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    packed_half_136 = (matrix_accumulator_62 << 0x5u) + 0x7ff88000u;

    packed_half_134 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    packed_half_117 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                        select_half_operand<2, false, false>(packed_half_128));

    packed_half_140 = (matrix_accumulator_119 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                        select_half_operand<2, false, false>(packed_half_148));

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                                       select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                        select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_3),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_109),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    packed_half_137 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_135 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                        select_half_operand<2, false, false>(packed_half_140));

    packed_half_115 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    packed_half_116 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3c203c20u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3c203c20u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x3c203c20u);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_5),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109), 0x3c203c20u);

    matrix_accumulator_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x3c203c20u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3c203c20u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                        select_half_operand<2, false, false>(packed_half_135));

    packed_half_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3e473e47u);

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                        select_half_operand<2, false, false>(packed_half_116));

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3e473e47u);

    condition_6 = (packed_half_0 & 0x1u) != 0;

    condition_4 = (packed_half_0 & 0x2u) != 0;

    condition_0 = (packed_half_0 & 0x8u) != 0;

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_68, matrix_fragment_69, 0u, 0u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0x3c203c20u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4c70ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3c203c20u);

    matrix_fragment_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x3e473e47u);

    matrix_fragment_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_109), 0x3e473e47u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0x3c203c20u);

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x3e473e47u);

    packed_half_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3e473e47u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0x3e473e47u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x3e473e47u);

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0x3e473e47u);

    matrix_fragment_16 = (packed_half_104 << 0x5u) + 0x7ff88000u;

    matrix_fragment_111 = (packed_half_106 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    packed_half_147 = (matrix_fragment_7 << 0x5u) + 0x7ff88000u;

    packed_half_146 = (matrix_fragment_109 << 0x5u) + 0x7ff88000u;

    packed_half_110 = (packed_half_105 << 0x5u) + 0x7ff88000u;

    packed_half_114 = (packed_half_107 << 0x5u) + 0x7ff88000u;

    packed_half_113 = (packed_half_112 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = (matrix_fragment_5 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_111));

    packed_half_112 = (matrix_fragment_4 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                       select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_17 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                        select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_4 = packed_half_0 * 0x4u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_61 = matrix_fragment_4 & 0x1cu;

    matrix_fragment_4 = condition_6 ? matrix_fragment_5 : matrix_fragment_12;

    matrix_fragment_7 = condition_6 ? matrix_accumulator_3 : matrix_fragment_6;

    matrix_fragment_5 = condition_6 ? matrix_fragment_12 : matrix_fragment_5;

    matrix_fragment_6 = condition_6 ? matrix_fragment_6 : matrix_accumulator_3;

    matrix_accumulator_2 =
        uint32_t(int64_t(int32_t(packed_half_0)) * (1ll << 0x1du) >> 32) + matrix_accumulator_61;

    multiply_accumulate_half(matrix_fragment_108, matrix_fragment_109, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_70, matrix_fragment_71, 0u, 0u);

    matrix_accumulator_61 = !condition_4 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_12 = !condition_4 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_accumulator_63 = !condition_4 ? matrix_fragment_5 : matrix_fragment_6;

    packed_half_105 = !condition_4 ? matrix_fragment_6 : matrix_fragment_5;

    matrix_fragment_4 = 0x1u ^ matrix_accumulator_2;

    matrix_fragment_7 = load_aligned_word<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x5070ull, memory_ranges, error_counts);

    matrix_fragment_5 = 0x2u ^ matrix_accumulator_2;

    matrix_fragment_6 = 0x3u ^ matrix_accumulator_2;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_61), "r"(matrix_accumulator_2), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_61 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_63), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_12 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_105), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        packed_half_105 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    condition_1 = (packed_half_0 & 0x10u) != 0;

    matrix_accumulator_3 = load_aligned_word<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x5070ull, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_8 = load_aligned_word<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x5070ull, memory_ranges, error_counts);

    matrix_fragment_9 = !condition_0 ? matrix_accumulator_61 : matrix_fragment_14;

    matrix_fragment_10 = !condition_0 ? matrix_fragment_12 : packed_half_105;

    matrix_fragment_11 = !condition_0 ? packed_half_105 : matrix_fragment_12;

    matrix_fragment_12 = !condition_1 ? matrix_fragment_9 : matrix_fragment_10;

    matrix_fragment_10 = !condition_1 ? matrix_fragment_10 : matrix_fragment_9;

    matrix_fragment_9 = load_aligned_word<CheckBounds>(
        combine_words(address_word_156, packed_half_157) + 0x5070ull, memory_ranges, error_counts);

    matrix_fragment_14 = !condition_0 ? matrix_fragment_14 : matrix_accumulator_61;

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4e70ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    packed_half_105 = !condition_1 ? matrix_fragment_14 : matrix_fragment_11;

    matrix_fragment_11 = !condition_1 ? matrix_fragment_11 : matrix_fragment_14;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_12 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_12),
                                        select_half_operand<1, false, false>(matrix_fragment_12));

    matrix_fragment_12 = permute_signed(matrix_fragment_12, matrix_fragment_12, 0x5410u);

    matrix_fragment_10 = packed_half_0;

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_68, matrix_fragment_69, 0u, 0u);

    matrix_fragment_11 = uint32_t(int32_t(matrix_fragment_10) >> 0x2u);

    matrix_fragment_14 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    matrix_fragment_14 = reciprocal_bits(matrix_fragment_14);

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_68, matrix_fragment_69, 0u, 0u);

    matrix_fragment_68 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_fragment_12 = matrix_fragment_11 + 0x8u;

    matrix_accumulator_139 = reciprocal_bits(matrix_fragment_68);

    matrix_fragment_69 = uint32_t(int32_t(matrix_fragment_12) >> 0x1fu);

    matrix_fragment_69 =
        uint32_t(combine_words(matrix_fragment_69, 0u) << 0x5u >> 32) + matrix_fragment_12;

    matrix_fragment_69 = matrix_fragment_69 & 0xffffffe0u;

    address_word_156 = matrix_fragment_12 + (0u - matrix_fragment_69);

    matrix_accumulator_139 = pack_half(matrix_accumulator_139, matrix_fragment_14);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_139), "r"(address_word_156), "r"(0x1fu), "r"(__activemask()));
        address_word_156 = result_word;
    }

    multiply_accumulate_half(packed_half_130, packed_half_131, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_70, matrix_fragment_71, 0u, 0u);

    matrix_fragment_14 = uint32_t(int32_t(matrix_fragment_10) >> 0x1fu);

    matrix_accumulator_123 = transpose_half_tile(matrix_accumulator_123);

    matrix_fragment_14 =
        uint32_t(combine_words(matrix_fragment_14, 0u) << 0x5u >> 32) + matrix_fragment_11;

    matrix_fragment_69 = transpose_half_tile(matrix_fragment_108);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_70, matrix_fragment_71, 0u, 0u);

    matrix_fragment_70 = transpose_half_tile(matrix_fragment_109);

    packed_half_122 = transpose_half_tile(packed_half_122);

    matrix_fragment_14 = matrix_fragment_14 & 0xffffffe0u;

    packed_half_157 = permute_signed(address_word_156, address_word_156, 0x5410u);

    matrix_fragment_14 = matrix_fragment_11 + (0u - matrix_fragment_14);

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(packed_half_157));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_139), "r"(matrix_fragment_14), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_68 = result_word;
    }

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                      select_half_operand<2, false, false>(0u));

    packed_half_128 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    address_word_156 = transpose_half_tile(packed_half_121);

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_108, matrix_fragment_109, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                       select_half_operand<2, false, false>(packed_half_128));

    multiply_accumulate_half(packed_half_122, matrix_accumulator_123, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_47 = transpose_half_tile(packed_half_120);

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                          select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_46 = permute_signed(matrix_fragment_68, matrix_fragment_68, 0x5410u);

    matrix_fragment_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                             select_half_operand<2, false, false>(packed_half_157));

    packed_half_148 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(0u));

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                          select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_150 = transpose_half_tile(matrix_fragment_99);

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                          select_half_operand<2, false, false>(matrix_fragment_46));

    address_word_156 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_156),
                                     select_half_operand<2, false, false>(0u));

    packed_half_121 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_152 = transpose_half_tile(matrix_accumulator_100);

    packed_half_149 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_98 = transpose_half_tile(matrix_fragment_98);

    matrix_accumulator_100 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                           select_half_operand<2, false, false>(address_word_156));

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    matrix_fragment_47 = transpose_half_tile(matrix_fragment_37);

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(packed_half_149));

    packed_half_151 = transpose_half_tile(matrix_accumulator_101);

    packed_half_148 = transpose_half_tile(matrix_accumulator_125);

    packed_half_131 = transpose_half_tile(packed_half_131);

    matrix_fragment_99 = transpose_half_tile(matrix_fragment_36);

    matrix_accumulator_126 = transpose_half_tile(matrix_accumulator_126);

    packed_half_149 = transpose_half_tile(matrix_accumulator_124);

    packed_half_130 = transpose_half_tile(packed_half_130);

    matrix_accumulator_133 = transpose_half_tile(matrix_accumulator_133);

    matrix_accumulator_132 = transpose_half_tile(matrix_accumulator_132);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    packed_half_151 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                    select_half_operand<2, false, false>(0u));

    packed_half_148 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(0u));

    packed_half_142 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                          select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                       select_half_operand<2, false, false>(0u));

    packed_half_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                          select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    packed_half_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                          select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(packed_half_121));

    packed_half_157 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                          select_half_operand<2, false, false>(packed_half_157));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                       select_half_operand<2, false, false>(packed_half_151));

    packed_half_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                          select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(packed_half_148));

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(0u));

    packed_half_105 = transpose_half_tile(packed_half_105);

    matrix_accumulator_133 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                           select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_140 = transpose_half_tile(matrix_accumulator_118);

    packed_half_130 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                    select_half_operand<2, false, false>(0u));

    packed_half_141 = transpose_half_tile(matrix_accumulator_119);

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    packed_half_131 = transpose_half_tile(packed_half_104);

    matrix_accumulator_132 = transpose_half_tile(matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_100, matrix_accumulator_101, 0u, 0u);

    packed_half_122 = transpose_half_tile(packed_half_122);

    packed_half_128 = transpose_half_tile(packed_half_128);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_109 = transpose_half_tile(matrix_fragment_109);

    matrix_accumulator_123 = transpose_half_tile(packed_half_106);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_98, matrix_fragment_99, 0u, 0u);

    matrix_fragment_69 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                      select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_133 = transpose_half_tile(packed_half_129);

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                       select_half_operand<2, false, false>(packed_half_130));

    packed_half_130 = transpose_half_tile(packed_half_107);

    packed_half_129 = transpose_half_tile(matrix_fragment_108);

    packed_half_142 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(0u));

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_131),
                                    select_half_operand<2, false, false>(packed_half_142));

    packed_half_131 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    packed_half_141 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                                    select_half_operand<2, false, false>(0u));

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                                    select_half_operand<2, false, false>(packed_half_131));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                    select_half_operand<2, false, false>(matrix_accumulator_133));

    packed_half_130 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_130),
                                    select_half_operand<2, false, false>(0u));

    packed_half_122 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_109),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_46, matrix_fragment_47, packed_half_120,
                             packed_half_121);

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                    select_half_operand<2, false, false>(packed_half_141));

    matrix_fragment_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                      select_half_operand<2, false, false>(packed_half_130));

    matrix_fragment_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                        select_half_operand<2, false, false>(packed_half_122));

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_106, packed_half_107, matrix_accumulator_118,
                             matrix_accumulator_119);

    uniform_index_word_5 = blockIdx.y;

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_104, packed_half_105, matrix_accumulator_126,
                             matrix_accumulator_127);

    uniform_block_index_9 = blockIdx.x;

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_108, matrix_fragment_109, matrix_accumulator_124,
                             matrix_accumulator_125);

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_120),
                                       select_half_operand<2, false, false>(0u));

    packed_half_120 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_121),
                                    select_half_operand<2, false, false>(0u));

    packed_half_121 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    uniform_parameter_word_8 = (uniform_index_word_5 << 0x3u) + uniform_parameter_word_11;

    uniform_block_index_9 = (uniform_block_index_9 << 0x3u) + uniform_parameter_word_10;

    uniform_condition_0 = int32_t(uniform_index_word_6) > int32_t(0u);

    uniform_condition_1 = int32_t(uniform_parameter_word_7) > int32_t(0u);

    matrix_accumulator_119 = packed_half_0;

    uniform_index_word_5 = uniform_condition_0 ? uniform_index_word_6 : uniform_parameter_word_12;

    matrix_fragment_71 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                      select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_accumulator_118 =
        uint32_t(combine_words(matrix_fragment_11, 0u) << 0x2u >> 32) + matrix_fragment_11;

    uniform_index_word_6 =
        uniform_condition_1 ? uniform_parameter_word_7 : uniform_parameter_word_13;

    matrix_accumulator_118 = matrix_accumulator_118 & 0xfffffffcu;

    matrix_accumulator_119 =
        uint32_t(int64_t(int32_t(matrix_accumulator_119)) * (1ll << 0x1cu) >> 32) +
        uniform_parameter_word_8;

    packed_half_122 = matrix_fragment_11 + uniform_block_index_9 + (0u - matrix_accumulator_118);

    condition_2 = int32_t(matrix_accumulator_119) >= int32_t(uniform_index_word_5);

    matrix_accumulator_123 = matrix_accumulator_119 + 0x2u;

    condition_3 = int32_t(packed_half_122) >= int32_t(uniform_index_word_6);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    condition_2 = int32_t(matrix_accumulator_119) < int32_t(0u) || condition_2;

    condition_4 = int32_t(matrix_accumulator_123) >= int32_t(uniform_index_word_5);

    condition_3 = int32_t(packed_half_122) < int32_t(0u) || condition_3;

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(packed_half_120));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(packed_half_121));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    condition_5 = condition_2 | condition_3;

    matrix_accumulator_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    condition_4 = int32_t(matrix_accumulator_123) < int32_t(0u) || condition_4;

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_90,
                             matrix_accumulator_91);

    condition_6 = condition_4 | condition_3;

    if (!condition_5) {
        packed_half_120 = parameters[parameter_output_low_2];
        packed_half_121 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_88,
                             matrix_accumulator_89);

    uniform_parameter_word_7 = uniform_index_word_6 << (0x2u & 31u);

    if (!condition_5) {
        matrix_accumulator_118 = matrix_accumulator_119 + uniform_index_word_5;
    }

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_84,
                             matrix_accumulator_85);

    if (!condition_5) {
        packed_half_129 = packed_half_122 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_86,
                             matrix_accumulator_87);

    if (!condition_6) {
        matrix_fragment_70 = parameters[parameter_output_low_2];
        matrix_fragment_71 = parameters[parameter_output_high_3];
    }

    if (!condition_5) {
        matrix_accumulator_124 = packed_half_122 * 0x4u;

        packed_half_129 = matrix_accumulator_118 * uniform_parameter_word_7 + packed_half_129;
    }

    matrix_accumulator_118 = matrix_fragment_10 & 0xfffffffcu;

    if (!condition_5) {
        matrix_accumulator_125 =
            matrix_accumulator_119 * uniform_parameter_word_7 + matrix_accumulator_124;
    }

    if (!condition_6) {
        matrix_accumulator_126 = packed_half_122 * 0x4u;
    }

    packed_half_131 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                    select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_accumulator_86 = parameters[parameter_output_low_2];
        matrix_accumulator_87 = parameters[parameter_output_high_3];

        matrix_accumulator_125 =
            matrix_accumulator_125 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_6) {
        matrix_accumulator_127 =
            matrix_accumulator_123 * uniform_parameter_word_7 + matrix_accumulator_126;
    }

    matrix_accumulator_90 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_91 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                      select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        split_address_words(packed_half_120, packed_half_121,
                            int64_t(int32_t(matrix_accumulator_125)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_120, packed_half_121));

        matrix_accumulator_89 = packed_half_129 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    matrix_accumulator_88 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                      select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_accumulator_127 =
            matrix_accumulator_127 + packed_half_0 + (0u - matrix_accumulator_118);

        matrix_accumulator_125 =
            permute_signed(matrix_accumulator_90, matrix_accumulator_91, 0x5410u);

        matrix_accumulator_91 = packed_half_122 * 0x4u;
    }

    packed_half_129 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                    select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_fragment_68 = matrix_accumulator_123 + uniform_index_word_5;
    }

    if (!condition_5) {
        packed_half_131 = permute_signed(matrix_accumulator_88, packed_half_131, 0x5410u);
    }

    if (!condition_6) {
        split_address_words(matrix_fragment_70, matrix_fragment_71,
                            int64_t(int32_t(matrix_accumulator_127)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_70, matrix_fragment_71));
    }

    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                      select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_accumulator_91 =
            matrix_fragment_68 * uniform_parameter_word_7 + matrix_accumulator_91;
    }

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        pipeline_write<CheckBounds>(combine_words(packed_half_120, packed_half_121),
                                    packed_half_131, memory_ranges, error_counts);
    }

    if (!condition_6) {
        matrix_accumulator_85 = permute_signed(matrix_accumulator_85, matrix_fragment_68, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_70, matrix_fragment_71),
                                    matrix_accumulator_125, memory_ranges, error_counts);
    }

    matrix_fragment_68 =
        uint32_t(int64_t(int32_t(matrix_fragment_10)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_69 = uint32_t(int32_t(matrix_fragment_68) >> 0x1fu);

    matrix_fragment_70 =
        uint32_t(int64_t(int32_t(matrix_fragment_10)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_71 = uint32_t(int32_t(matrix_fragment_70) >> 0x1fu);

    matrix_fragment_69 =
        uint32_t(combine_words(matrix_fragment_69, 0u) << 0x5u >> 32) + matrix_fragment_68;

    matrix_fragment_71 =
        uint32_t(combine_words(matrix_fragment_71, 0u) << 0x5u >> 32) + matrix_fragment_70;

    matrix_fragment_69 = matrix_fragment_69 & 0xffffffe0u;

    matrix_fragment_71 = matrix_fragment_71 & 0xffffffe0u;

    matrix_fragment_68 = matrix_fragment_68 + (0u - matrix_fragment_69);

    matrix_fragment_70 = matrix_fragment_70 + (0u - matrix_fragment_71);

    if (!condition_5) {
        split_address_words(matrix_accumulator_88, matrix_accumulator_89,
                            int64_t(int32_t(matrix_accumulator_89)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_86, matrix_accumulator_87));
    }

    if (!condition_6) {
        matrix_accumulator_86 = parameters[parameter_output_low_2];
        matrix_accumulator_87 = parameters[parameter_output_high_3];
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_139), "r"(matrix_fragment_68), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_68 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_139), "r"(matrix_fragment_70), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                      select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_accumulator_91 =
            matrix_accumulator_91 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_5) {
        matrix_accumulator_127 = permute_signed(matrix_accumulator_84, packed_half_129, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_88, matrix_accumulator_89),
                                    matrix_accumulator_127, memory_ranges, error_counts);
    }

    packed_half_120 = permute_signed(matrix_fragment_68, matrix_fragment_68, 0x5410u);

    packed_half_121 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5410u);

    if (!condition_6) {
        split_address_words(matrix_accumulator_86, matrix_accumulator_87,
                            int64_t(int32_t(matrix_accumulator_91)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_86, matrix_accumulator_87));
    }

    packed_half_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                          select_half_operand<2, false, false>(packed_half_120));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                          select_half_operand<2, false, false>(packed_half_120));

    packed_half_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<2, false, false>(packed_half_121));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                          select_half_operand<2, false, false>(packed_half_121));

    if (!condition_6) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_86, matrix_accumulator_87),
                                    matrix_accumulator_85, memory_ranges, error_counts);
    }

    packed_half_138 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                                          select_half_operand<2, false, false>(packed_half_120));

    packed_half_117 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                    select_half_operand<2, false, false>(0u));

    packed_half_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                          select_half_operand<2, false, false>(packed_half_120));

    matrix_accumulator_84 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                             select_half_operand<2, false, false>(packed_half_121));

    packed_half_116 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                             select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_134),
                                       select_half_operand<2, false, false>(packed_half_117));

    matrix_accumulator_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_138),
                                       select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(packed_half_116));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(packed_half_120));

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                          select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(packed_half_121));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_100, matrix_accumulator_101, 0u, 0u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_111 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_111),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    packed_half_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_98, matrix_fragment_99, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_70 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                             select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                             select_half_operand<2, false, false>(packed_half_121));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                             select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_111));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(packed_half_110));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_104, packed_half_105, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_106, packed_half_107, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_108, matrix_fragment_109, matrix_accumulator_90,
                             matrix_accumulator_91);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = packed_half_122 + 0x4u;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                       select_half_operand<2, false, false>(0u));

    condition_5 = int32_t(matrix_fragment_15) >= int32_t(uniform_index_word_6);

    matrix_accumulator_86 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    condition_5 = int32_t(matrix_fragment_15) < int32_t(0u) || condition_5;

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(matrix_accumulator_86));

    condition_4 = condition_4 | condition_5;

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_72,
                             matrix_accumulator_73);

    condition_2 = condition_2 | condition_5;

    if (!condition_4) {
        matrix_accumulator_84 = parameters[parameter_output_low_2];
        matrix_accumulator_85 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_80,
                             matrix_accumulator_81);

    if (!condition_2) {
        matrix_fragment_16 = parameters[parameter_output_low_2];
        matrix_fragment_17 = parameters[parameter_output_high_3];

        matrix_accumulator_66 = parameters[parameter_output_low_2];
        matrix_accumulator_67 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_82,
                             matrix_accumulator_83);

    if (!condition_4) {
        matrix_accumulator_64 = parameters[parameter_output_low_2];
        matrix_accumulator_65 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_accumulator_74 = matrix_accumulator_119 + uniform_index_word_5;
    }

    if (!condition_4) {
        matrix_fragment_70 = matrix_fragment_15 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_68 = matrix_fragment_15 * 0x4u;
    }

    if (!condition_4) {
        matrix_fragment_69 = matrix_accumulator_123 * uniform_parameter_word_7 + matrix_fragment_70;
    }

    if (!condition_2) {
        matrix_accumulator_119 =
            matrix_accumulator_119 * uniform_parameter_word_7 + matrix_fragment_68;
    }

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_fragment_69 = matrix_fragment_69 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_73 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_71 = matrix_fragment_15 * 0x4u;

        matrix_accumulator_73 =
            permute_signed(matrix_accumulator_60, matrix_accumulator_73, 0x5410u);
    }

    if (!condition_4) {
        split_address_words(matrix_accumulator_64, matrix_accumulator_65,
                            int64_t(int32_t(matrix_fragment_69)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_64, matrix_accumulator_65));

        matrix_accumulator_60 = matrix_accumulator_123 + uniform_index_word_5;

        matrix_fragment_69 = matrix_fragment_15 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_71 = matrix_accumulator_74 * uniform_parameter_word_7 + matrix_fragment_71;
    }

    if (!condition_4) {
        matrix_fragment_69 = matrix_accumulator_60 * uniform_parameter_word_7 + matrix_fragment_69;
    }

    if (!condition_2) {
        matrix_accumulator_119 =
            matrix_accumulator_119 + packed_half_0 + (0u - matrix_accumulator_118);

        matrix_fragment_71 = matrix_fragment_71 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    matrix_accumulator_75 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_72 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                      select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_fragment_69 = matrix_fragment_69 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_accumulator_119)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        split_address_words(matrix_accumulator_66, matrix_accumulator_67,
                            int64_t(int32_t(matrix_fragment_71)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_66, matrix_accumulator_67));

        matrix_accumulator_75 =
            permute_signed(matrix_accumulator_75, matrix_accumulator_72, 0x5410u);
    }

    if (!condition_4) {
        matrix_fragment_71 = permute_signed(matrix_accumulator_83, matrix_fragment_68, 0x5410u);

        split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                            int64_t(int32_t(matrix_fragment_69)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_84, matrix_accumulator_85));

        matrix_accumulator_87 =
            permute_signed(matrix_accumulator_61, matrix_accumulator_60, 0x5410u);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                    matrix_accumulator_75, memory_ranges, error_counts);
    }

    if (!condition_4) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_64, matrix_accumulator_65),
                                    matrix_fragment_71, memory_ranges, error_counts);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_66, matrix_accumulator_67),
                                    matrix_accumulator_73, memory_ranges, error_counts);
    }

    if (!condition_4) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_84, matrix_accumulator_85),
                                    matrix_accumulator_87, memory_ranges, error_counts);
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3c60ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x3e60ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4060ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4260ull, matrix_accumulator_72,
        matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    multiply_accumulate_half(packed_half_110, matrix_fragment_111, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4660ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4860ull, matrix_accumulator_84,
        matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4a60ull, matrix_accumulator_88,
        matrix_accumulator_89, matrix_accumulator_90, matrix_accumulator_91, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(packed_half_114, packed_half_115, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4460ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    packed_half_120 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                          select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_116 = multiply_half_pairs(select_half_operand<1, false, false>(matrix_fragment_17),
                                          select_half_operand<1, false, false>(matrix_fragment_17));

    packed_half_121 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                      select_half_operand<2, false, false>(packed_half_110),
                                      select_half_operand<2, false, false>(packed_half_120));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<2, false, false>(packed_half_112));

    packed_half_120 = multiply_half_pairs(select_half_operand<0, false, false>(matrix_fragment_17),
                                          select_half_operand<0, false, false>(matrix_fragment_17));

    matrix_accumulator_119 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                      select_half_operand<2, false, false>(packed_half_114),
                                      select_half_operand<2, false, false>(packed_half_117));

    packed_half_117 =
        fused_multiply_add_half_pairs(select_half_operand<1, false, false>(matrix_fragment_111),
                                      select_half_operand<1, false, false>(matrix_fragment_111),
                                      select_half_operand<0, false, false>(packed_half_116));

    matrix_accumulator_119 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                       select_half_operand<2, false, false>(packed_half_121));

    packed_half_121 =
        fused_multiply_add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_111),
                                      select_half_operand<0, false, false>(matrix_fragment_111),
                                      select_half_operand<0, false, false>(packed_half_120));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                          select_half_operand<2, false, false>(packed_half_113));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_120 = result_word;
    }

    packed_half_121 = permute_signed(packed_half_121, packed_half_117, 0x5410u);

    packed_half_116 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                      select_half_operand<2, false, false>(packed_half_115),
                                      select_half_operand<2, false, false>(packed_half_116));

    packed_half_116 = add_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                     select_half_operand<2, false, false>(packed_half_121));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_116), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_121 = result_word;
    }

    packed_half_120 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                     select_half_operand<2, false, false>(packed_half_120));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_120), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_117 = result_word;
    }

    packed_half_121 = add_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                     select_half_operand<2, false, false>(packed_half_121));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_121), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_122 = result_word;
    }

    packed_half_117 = add_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                     select_half_operand<2, false, false>(packed_half_117));

    packed_half_116 = permute_signed(packed_half_117, packed_half_117, 0x5432u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    packed_half_117 = add_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                     select_half_operand<2, false, false>(packed_half_116));

    packed_half_122 = add_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                                     select_half_operand<2, false, false>(packed_half_122));

    packed_half_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_117), 0x4100410u);

    matrix_accumulator_119 = permute_signed(packed_half_122, packed_half_122, 0x5432u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    packed_half_122 = add_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                     select_half_operand<2, false, false>(matrix_accumulator_119));

    packed_half_116 = half_to_float_bits((packed_half_117 & 65535u) * 65537u);

    packed_half_117 = half_to_float_bits((packed_half_117 >> 16) * 65537u);

    packed_half_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_122), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    packed_half_116 = rsqrt_bits(packed_half_116);

    matrix_accumulator_119 = half_to_float_bits((packed_half_122 & 65535u) * 65537u);

    packed_half_122 = half_to_float_bits((packed_half_122 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    packed_half_117 = rsqrt_bits(packed_half_117);

    matrix_accumulator_119 = rsqrt_bits(matrix_accumulator_119);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    packed_half_122 = rsqrt_bits(packed_half_122);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    packed_half_120 = pack_half(packed_half_117, packed_half_116);

    matrix_fragment_48 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_49 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_50 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_51 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_51));

    packed_half_116 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                     select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(packed_half_120));

    packed_half_117 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                     select_half_operand<2, false, false>(matrix_fragment_49));

    packed_half_122 = pack_half(packed_half_122, matrix_accumulator_119);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(packed_half_13));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_116), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_119 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_117), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                          select_half_operand<2, false, false>(packed_half_120));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<2, false, false>(packed_half_120));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                          select_half_operand<2, false, false>(packed_half_122));

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                          select_half_operand<2, false, false>(packed_half_122));

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                          select_half_operand<0, false, false>(packed_half_13));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<0, false, false>(packed_half_13));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(packed_half_122));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<2, false, false>(packed_half_122));

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(0u));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(packed_half_13));

    packed_half_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                            select_half_operand<0, false, false>(packed_half_13));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                    select_half_operand<2, false, false>(0u));

    packed_half_116 = add_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                     select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_111),
                                       select_half_operand<2, false, false>(packed_half_112));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_116), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_111 = result_word;
    }

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_111 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                         select_half_operand<2, false, false>(matrix_fragment_111));

    packed_half_110 = permute_signed(matrix_fragment_17, matrix_fragment_17, 0x5432u);

    matrix_fragment_16 = permute_signed(matrix_fragment_111, matrix_fragment_111, 0x5432u);

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_111 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111),
                                         select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x4100410u);

    matrix_fragment_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_111), 0x4100410u);

    packed_half_110 = half_to_float_bits((matrix_fragment_17 & 65535u) * 65537u);

    matrix_fragment_17 = half_to_float_bits((matrix_fragment_17 >> 16) * 65537u);

    matrix_fragment_16 = half_to_float_bits((matrix_fragment_111 & 65535u) * 65537u);

    matrix_fragment_111 = half_to_float_bits((matrix_fragment_111 >> 16) * 65537u);

    packed_half_110 = rsqrt_bits(packed_half_110);

    matrix_fragment_17 = rsqrt_bits(matrix_fragment_17);

    matrix_fragment_16 = rsqrt_bits(matrix_fragment_16);

    matrix_fragment_111 = rsqrt_bits(matrix_fragment_111);

    packed_half_110 = pack_half(matrix_fragment_17, packed_half_110);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(packed_half_110));

    packed_half_112 = pack_half(matrix_fragment_111, matrix_fragment_16);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<0, false, false>(packed_half_13));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_96, matrix_fragment_97, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<0, false, false>(packed_half_13));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(packed_half_13));

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             packed_half_102, packed_half_103, matrix_fragment_70,
                             matrix_fragment_71);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_96, matrix_fragment_97, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_62 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3c203c20u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3e473e47u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_34 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = (matrix_accumulator_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_54 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_16 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_30 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_35 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_52 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_55 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_59 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_61 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_60 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             packed_half_102, packed_half_103, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_41 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_43 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                       select_half_operand<2, false, false>(matrix_accumulator_60));

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_90,
                             matrix_accumulator_91);

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4c70ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_accumulator_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_154, address_word_155) + 0x4e70ull, packed_half_152,
        packed_half_153, address_word_154, address_word_155, memory_ranges, error_counts);

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_accumulator_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    matrix_accumulator_84 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x3c203c20u);

    matrix_accumulator_86 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3c203c20u);

    matrix_accumulator_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_84 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x3e473e47u);

    matrix_accumulator_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_86 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3e473e47u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_64 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_65 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_66 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_67 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_85 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85), 0x3c203c20u);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91),
        select_half_operand<1, false, false>(packed_half_13), 0x3d343d34u);

    matrix_accumulator_87 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3c203c20u);

    matrix_fragment_51 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_accumulator_72 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x3c203c20u);

    matrix_fragment_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                       select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_accumulator_62 = (matrix_accumulator_84 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_63 = (matrix_accumulator_86 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_73 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3c203c20u);

    matrix_accumulator_85 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85), 0x3e473e47u);

    matrix_accumulator_87 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3e473e47u);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3c203c20u);

    matrix_accumulator_75 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3c203c20u);

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3c203c20u);

    matrix_accumulator_89 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3c203c20u);

    matrix_accumulator_90 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3c203c20u);

    matrix_accumulator_91 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3c203c20u);

    matrix_fragment_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_72 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x3e473e47u);

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_accumulator_73 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3e473e47u);

    matrix_accumulator_74 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3e473e47u);

    matrix_accumulator_75 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3e473e47u);

    matrix_accumulator_88 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3e473e47u);

    matrix_accumulator_89 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3e473e47u);

    matrix_accumulator_90 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3e473e47u);

    matrix_accumulator_91 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3e473e47u);

    packed_half_13 = (matrix_accumulator_85 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = (matrix_accumulator_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_69 = (matrix_accumulator_72 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_72 = (matrix_accumulator_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_74 = (matrix_accumulator_74 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_75 = (matrix_accumulator_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = (matrix_accumulator_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = (matrix_accumulator_89 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_73 = (matrix_accumulator_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = (matrix_accumulator_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_49 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_fragment_51 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                       select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_78 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                       select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_fragment_76));

    condition_2 = (packed_half_0 & 0x1u) != 0;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_49 = condition_2 ? matrix_fragment_51 : matrix_fragment_48;

    condition_6 = (packed_half_0 & 0x2u) != 0;

    matrix_fragment_50 = condition_2 ? matrix_accumulator_80 : matrix_fragment_77;

    matrix_fragment_48 = condition_2 ? matrix_fragment_48 : matrix_fragment_51;

    matrix_fragment_77 = condition_2 ? matrix_fragment_77 : matrix_accumulator_80;

    matrix_fragment_51 = !condition_6 ? matrix_fragment_49 : matrix_fragment_50;

    matrix_fragment_50 = !condition_6 ? matrix_fragment_50 : matrix_fragment_49;

    matrix_fragment_49 = !condition_6 ? matrix_fragment_48 : matrix_fragment_77;

    matrix_fragment_77 = !condition_6 ? matrix_fragment_77 : matrix_fragment_48;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_51), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_48 = !condition_0 ? matrix_fragment_51 : matrix_fragment_4;

    matrix_fragment_78 = !condition_0 ? matrix_fragment_4 : matrix_fragment_51;

    matrix_fragment_79 = !condition_0 ? matrix_fragment_50 : matrix_fragment_77;

    matrix_accumulator_81 = !condition_0 ? matrix_fragment_77 : matrix_fragment_50;

    matrix_accumulator_80 = !condition_1 ? matrix_fragment_48 : matrix_fragment_79;

    matrix_accumulator_83 = !condition_1 ? matrix_fragment_78 : matrix_accumulator_81;

    matrix_fragment_48 = !condition_1 ? matrix_fragment_79 : matrix_fragment_48;

    matrix_fragment_78 = !condition_1 ? matrix_accumulator_81 : matrix_fragment_78;

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_80),
                       select_half_operand<1, false, false>(matrix_accumulator_80));

    matrix_accumulator_80 = permute_signed(matrix_accumulator_80, matrix_accumulator_80, 0x5410u);

    matrix_accumulator_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80), 0x4100410u);

    matrix_accumulator_2 = half_to_float_bits((matrix_accumulator_80 & 65535u) * 65537u);

    matrix_accumulator_80 = half_to_float_bits((matrix_accumulator_80 >> 16) * 65537u);

    matrix_fragment_49 = uint32_t(int32_t(matrix_fragment_12) >> 0x1fu);

    matrix_accumulator_2 = reciprocal_bits(matrix_accumulator_2);

    matrix_fragment_49 =
        uint32_t(combine_words(matrix_fragment_49, 0u) << 0x5u >> 32) + matrix_fragment_12;

    matrix_fragment_5 = reciprocal_bits(matrix_accumulator_80);

    matrix_fragment_49 = matrix_fragment_49 & 0xffffffe0u;

    matrix_fragment_6 =
        uint32_t(int64_t(int32_t(matrix_fragment_10)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_10 =
        uint32_t(int64_t(int32_t(matrix_fragment_10)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_4 = matrix_fragment_12 + (0u - matrix_fragment_49);

    matrix_fragment_49 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    matrix_fragment_51 = uint32_t(int32_t(matrix_fragment_10) >> 0x1fu);

    matrix_fragment_49 =
        uint32_t(combine_words(matrix_fragment_49, 0u) << 0x5u >> 32) + matrix_fragment_6;

    matrix_fragment_51 =
        uint32_t(combine_words(matrix_fragment_51, 0u) << 0x5u >> 32) + matrix_fragment_10;

    matrix_fragment_5 = pack_half(matrix_fragment_5, matrix_accumulator_2);

    matrix_fragment_49 = matrix_fragment_49 & 0xffffffe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    matrix_fragment_51 = matrix_fragment_51 & 0xffffffe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_fragment_6 = matrix_fragment_6 + (0u - matrix_fragment_49);

    matrix_fragment_10 = matrix_fragment_10 + (0u - matrix_fragment_51);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    matrix_fragment_14 = permute_signed(matrix_fragment_14, matrix_fragment_14, 0x5410u);

    matrix_accumulator_2 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5410u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5410u);

    matrix_fragment_10 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5410u);

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_100, matrix_accumulator_101, 0u, 0u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_98, matrix_fragment_99, 0u, 0u);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_98, matrix_fragment_99, 0u, 0u);

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_100, matrix_accumulator_101, 0u, 0u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                         select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_108, matrix_fragment_109, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             packed_half_104, packed_half_105, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                             matrix_fragment_45);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_106, packed_half_107, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_108, matrix_fragment_109, matrix_fragment_36,
                             matrix_fragment_37);

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                   select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             packed_half_106, packed_half_107, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_104, packed_half_105, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                      select_half_operand<2, false, false>(0u));

    uniform_index_word_6 = uniform_parameter_word_8 + 0x4u;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_4 = packed_half_0;

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        uint32_t(int64_t(int32_t(matrix_fragment_4)) * (1ll << 0x1cu) >> 32) + uniform_index_word_6;

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    condition_0 = int32_t(matrix_fragment_4) >= int32_t(uniform_index_word_5);

    matrix_fragment_14 = matrix_fragment_4 + 0x2u;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                      select_half_operand<2, false, false>(0u));

    condition_0 = int32_t(matrix_fragment_4) < int32_t(0u) || condition_0;

    condition_1 = int32_t(matrix_fragment_14) >= int32_t(uniform_index_word_5);

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_29 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_29),
                                         select_half_operand<2, false, false>(matrix_fragment_7));

    condition_2 = condition_0 | condition_3;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_11, 0u) << 0x2u >> 32) + matrix_fragment_11;

    condition_1 = int32_t(matrix_fragment_14) < int32_t(0u) || condition_1;

    matrix_fragment_6 = matrix_fragment_5 & 0xfffffffcu;

    condition_3 = condition_1 | condition_3;

    multiply_accumulate_half(matrix_fragment_12, packed_half_13, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             packed_half_13);

    matrix_fragment_11 = matrix_fragment_11 + uniform_block_index_9 + (0u - matrix_fragment_6);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    if (!condition_2) {
        matrix_fragment_5 = matrix_fragment_4 + uniform_index_word_5;
    }

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_38,
                             matrix_fragment_39);

    condition_0 = condition_0 | condition_5;

    if (!condition_2) {
        matrix_fragment_8 = matrix_fragment_11 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_28,
                             packed_half_29);

    if (!condition_2) {
        matrix_fragment_23 = matrix_fragment_5 * uniform_parameter_word_7 + matrix_fragment_8;
    }

    if (!condition_3) {
        matrix_fragment_5 = matrix_fragment_14 + uniform_index_word_5;
    }

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    condition_1 = condition_1 | condition_5;

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             packed_half_152, packed_half_153, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    if (!condition_0) {
        matrix_fragment_8 = matrix_fragment_4 + uniform_index_word_5;

        matrix_fragment_31 = matrix_fragment_15 * 0x4u;
    }

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_13),
                                   select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        matrix_fragment_16 = parameters[parameter_output_low_2];
        matrix_fragment_17 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(packed_half_152, packed_half_153, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             packed_half_152, packed_half_153, matrix_fragment_6,
                             matrix_fragment_7);

    if (!condition_3) {
        matrix_fragment_6 = matrix_fragment_11 * 0x4u;

        matrix_fragment_27 = matrix_fragment_5 * uniform_parameter_word_7 + matrix_fragment_6;
    }

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_accumulator_2 = parameters[parameter_output_low_2];
        matrix_accumulator_3 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             address_word_154, address_word_155, matrix_fragment_18,
                             matrix_fragment_19);

    if (!condition_0) {
        packed_half_29 = matrix_fragment_15 * 0x4u;
    }

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        matrix_fragment_31 = matrix_fragment_8 * uniform_parameter_word_7 + matrix_fragment_31;
    }

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_19 = matrix_fragment_11 * 0x4u;
    }

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        packed_half_29 = matrix_fragment_4 * uniform_parameter_word_7 + packed_half_29;
    }

    if (!condition_1) {
        matrix_fragment_43 = permute_signed(matrix_fragment_7, matrix_fragment_8, 0x5410u);
    }

    if (!condition_2) {
        matrix_fragment_19 = matrix_fragment_4 * uniform_parameter_word_7 + matrix_fragment_19;
    }

    if (!condition_3) {
        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             address_word_154, address_word_155, matrix_fragment_20,
                             matrix_fragment_21);

    if (!condition_2) {
        matrix_fragment_37 = permute_signed(matrix_fragment_12, matrix_fragment_37, 0x5410u);
    }

    if (!condition_3) {
        matrix_fragment_4 = parameters[parameter_output_low_2];
        matrix_fragment_5 = parameters[parameter_output_high_3];

        matrix_fragment_35 = permute_signed(packed_half_13, matrix_fragment_6, 0x5410u);

        matrix_fragment_21 = matrix_fragment_11 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_6 = parameters[parameter_output_low_2];
        matrix_fragment_7 = parameters[parameter_output_high_3];

        matrix_fragment_19 = matrix_fragment_19 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_0) {
        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];
    }

    if (!condition_1) {
        matrix_fragment_12 = parameters[parameter_output_low_2];
        packed_half_13 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (!condition_3) {
        matrix_fragment_21 = matrix_fragment_14 * uniform_parameter_word_7 + matrix_fragment_21;
    }

    if (!condition_1) {
        matrix_fragment_19 = matrix_fragment_15 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_23 = matrix_fragment_23 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_3) {
        matrix_fragment_21 = matrix_fragment_21 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_1) {
        matrix_fragment_19 = matrix_fragment_14 * uniform_parameter_word_7 + matrix_fragment_19;
    }

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_fragment_27 = matrix_fragment_27 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        packed_half_29 = packed_half_29 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_21)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_19 = matrix_fragment_19 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_0) {
        matrix_fragment_31 = matrix_fragment_31 + packed_half_0 + (0u - matrix_accumulator_118);
    }

    if (!condition_2) {
        matrix_fragment_39 = permute_signed(matrix_fragment_24, matrix_fragment_39, 0x5410u);
    }

    if (!condition_3) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    packed_half_152 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                    matrix_fragment_37, memory_ranges, error_counts);
    }

    if (!condition_3) {
        matrix_fragment_21 = permute_signed(matrix_fragment_25, matrix_fragment_18, 0x5410u);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(packed_half_29)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

        matrix_fragment_41 = permute_signed(matrix_fragment_40, matrix_fragment_41, 0x5410u);
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_fragment_35, memory_ranges, error_counts);
    }

    if (!condition_0) {
        matrix_fragment_25 = permute_signed(packed_half_152, matrix_fragment_45, 0x5410u);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_fragment_39, memory_ranges, error_counts);
    }

    if (!condition_1) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, packed_half_13));
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_21, memory_ranges, error_counts);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_12, packed_half_13,
                            int64_t(int32_t(matrix_fragment_31)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_fragment_41, memory_ranges, error_counts);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                    matrix_fragment_43, memory_ranges, error_counts);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_12, packed_half_13),
                                    matrix_fragment_25, memory_ranges, error_counts);
    }

    if (condition_1) {
        return;
    }

    matrix_accumulator_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    matrix_fragment_14 = matrix_fragment_14 + uniform_index_word_5;

    matrix_fragment_5 = matrix_fragment_15 * 0x4u;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = matrix_fragment_14 * uniform_parameter_word_7 + matrix_fragment_5;

    packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                    select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 = matrix_fragment_5 + packed_half_0 + (0u - matrix_accumulator_118);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                packed_half_153, memory_ranges, error_counts);
    return;
}
extern "C" __global__ __launch_bounds__(
    32, 13) void phoenix_window_attention_channels_32_output_view(KernelArguments parameters) {
    run_window_attention_channels_32_output_view<false>(parameters.words, nullptr, nullptr);
}
