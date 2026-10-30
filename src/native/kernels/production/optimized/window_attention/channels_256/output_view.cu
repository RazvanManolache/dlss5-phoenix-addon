#include "../../../common/architecture_policy.cuh"
// DLSS5 Phoenix optimized kernel: window_attention/channels_256/output_view.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (PHOENIX_STYLE_ARCH >= 900)
#define DLSS5_TILE_SHARED_BYTES 17408
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
__device__ __forceinline__ void run_window_attention_channels_256_output_view(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t tile_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t packed_half_3 = 0;
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
    uint32_t address_word_28 = 0;
    uint32_t index_word_29 = 0;
    uint32_t packed_half_30 = 0;
    uint32_t matrix_accumulator_31 = 0;
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
    uint32_t matrix_fragment_60 = 0;
    uint32_t matrix_fragment_61 = 0;
    uint32_t matrix_fragment_62 = 0;
    uint32_t matrix_fragment_63 = 0;
    uint32_t matrix_fragment_64 = 0;
    uint32_t matrix_fragment_65 = 0;
    uint32_t matrix_fragment_66 = 0;
    uint32_t matrix_fragment_67 = 0;
    uint32_t matrix_fragment_68 = 0;
    uint32_t matrix_fragment_69 = 0;
    uint32_t matrix_fragment_70 = 0;
    uint32_t matrix_fragment_71 = 0;
    uint32_t matrix_fragment_72 = 0;
    uint32_t matrix_fragment_73 = 0;
    uint32_t matrix_fragment_74 = 0;
    uint32_t matrix_fragment_75 = 0;
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t packed_half_78 = 0;
    uint32_t packed_half_79 = 0;
    uint32_t matrix_accumulator_80 = 0;
    uint32_t matrix_accumulator_81 = 0;
    uint32_t matrix_accumulator_82 = 0;
    uint32_t packed_half_83 = 0;
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
    uint32_t matrix_accumulator_100 = 0;
    uint32_t matrix_accumulator_101 = 0;
    uint32_t matrix_accumulator_102 = 0;
    uint32_t matrix_accumulator_103 = 0;
    uint32_t packed_half_104 = 0;
    uint32_t matrix_accumulator_105 = 0;
    uint32_t packed_half_106 = 0;
    uint32_t matrix_accumulator_107 = 0;
    uint32_t matrix_accumulator_108 = 0;
    uint32_t matrix_accumulator_109 = 0;
    uint32_t matrix_accumulator_110 = 0;
    uint32_t matrix_accumulator_111 = 0;
    uint32_t matrix_fragment_112 = 0;
    uint32_t matrix_fragment_113 = 0;
    uint32_t matrix_accumulator_114 = 0;
    uint32_t matrix_accumulator_115 = 0;
    uint32_t matrix_accumulator_116 = 0;
    uint32_t matrix_accumulator_117 = 0;
    uint32_t matrix_fragment_118 = 0;
    uint32_t matrix_fragment_119 = 0;
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
    uint32_t matrix_accumulator_130 = 0;
    uint32_t matrix_accumulator_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t matrix_accumulator_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t matrix_accumulator_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t matrix_accumulator_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t matrix_accumulator_144 = 0;
    uint32_t matrix_accumulator_145 = 0;
    uint32_t matrix_accumulator_146 = 0;
    uint32_t matrix_accumulator_147 = 0;
    uint32_t matrix_accumulator_148 = 0;
    uint32_t matrix_accumulator_149 = 0;
    uint32_t matrix_accumulator_150 = 0;
    uint32_t matrix_accumulator_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t address_word_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t packed_half_159 = 0;
    uint32_t address_word_160 = 0;
    uint32_t index_word_161 = 0;
    uint32_t address_word_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t index_word_164 = 0;
    uint32_t index_word_165 = 0;
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

    tile_word_0 = blockIdx.y;

    uniform_parameter_word_9 = parameters[parameter_optional_control_15];

    matrix_fragment_8 = parameters[parameter_optional_control_10];
    matrix_fragment_9 = parameters[parameter_scalar_control_11];

    matrix_fragment_5 = blockIdx.x;

    uniform_parameter_word_7 = parameters[parameter_tensor_height_8];

    matrix_fragment_6 = threadIdx.x;

    uniform_index_word_8 = parameters[parameter_scalar_control_14];

    packed_half_3 = threadIdx.y;

    uniform_parameter_word_6 = parameters[parameter_tensor_width_9];

    matrix_fragment_7 = threadIdx.z;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_9));

    uniform_parameter_word_7 = 0u - uniform_index_word_4 + uniform_parameter_word_7;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_8));

    uniform_index_word_4 = 0u;

    uniform_condition_1 = uint32_t(uniform_parameter_word_7) != 0u;

    uniform_parameter_word_6 = 0u - uniform_index_word_5 + uniform_parameter_word_6;

    tile_word_0 = tile_word_0 * 0x8u + matrix_fragment_9;

    uniform_index_word_5 = 0u;

    uniform_condition_0 = uint32_t(uniform_parameter_word_6) != 0u;

    address_word_2 = matrix_fragment_5 * 0x8u + matrix_fragment_8;

    matrix_fragment_4 = tile_word_0 + (0u - uniform_parameter_word_9);

    if (uniform_condition_1) {
        uniform_parameter_word_7 = uniform_parameter_word_7 + 0xffffffffu;
    }

    matrix_fragment_8 = address_word_2 + (0u - uniform_index_word_8);

    matrix_fragment_9 = matrix_fragment_4 + 0x8u;

    matrix_fragment_5 = matrix_fragment_4 + 0x7u;

    if (uniform_condition_1) {
        uniform_index_word_8 = uint32_t(int32_t(uniform_parameter_word_7) >> 0x1fu);
    }

    matrix_fragment_4 = max(int32_t(0u), int32_t(matrix_fragment_4));

    if (uniform_condition_0) {
        uniform_parameter_word_6 = uniform_parameter_word_6 + 0xffffffffu;
    }

    condition_1 = int32_t(matrix_fragment_9) > int32_t(0u);

    condition_0 = (matrix_fragment_6 | matrix_fragment_7 | packed_half_3) != 0;

    if (uniform_condition_1) {
        uniform_index_word_8 = uint32_t(combine_words(uniform_index_word_8, 0u) << 0x3u >> 32) +
                               uniform_parameter_word_7;
    }

    matrix_fragment_6 = uint32_t(combine_words(matrix_fragment_5, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_parameter_word_7 = uint32_t(int32_t(uniform_parameter_word_6) >> 0x1fu);
    }

    matrix_fragment_4 = uint32_t(uint32_t(matrix_fragment_4) >> 0x3u);

    if (uniform_condition_1) {
        uniform_index_word_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_8)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_fragment_10 = matrix_fragment_8 + 0x8u;

    matrix_fragment_6 = condition_1 ? matrix_fragment_6 : 0u;

    matrix_fragment_9 = matrix_fragment_8 + 0x7u;

    matrix_fragment_5 = uint32_t(int32_t(tile_word_0) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_parameter_word_7 =
            uint32_t(combine_words(uniform_parameter_word_7, 0u) << 0x3u >> 32) +
            uniform_parameter_word_6;
    }

    matrix_fragment_7 = uint32_t(int32_t(address_word_2) >> 0x1fu);

    matrix_fragment_11 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_5));

    condition_2 = int32_t(matrix_fragment_10) > int32_t(0u);

    matrix_fragment_9 = uint32_t(combine_words(matrix_fragment_9, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_fragment_5 = uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + tile_word_0;

    condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_11) || condition_0;

    matrix_fragment_7 =
        uint32_t(combine_words(matrix_fragment_7, 0u) << 0x2u >> 32) + address_word_2;

    matrix_fragment_8 = max(int32_t(0u), int32_t(matrix_fragment_8));

    matrix_fragment_12 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    matrix_fragment_13 = uint32_t(int32_t(matrix_fragment_7) >> 0x2u);

    matrix_fragment_9 = condition_2 ? matrix_fragment_9 : 0u;
    if (!condition_0) {
        matrix_fragment_8 = uint32_t(uint32_t(matrix_fragment_8) >> 0x3u);

        matrix_fragment_10 = min(int32_t(matrix_fragment_9), int32_t(uniform_index_word_4));

        matrix_fragment_6 = matrix_fragment_4;
        do {
            condition_0 = int32_t(matrix_fragment_8) >= int32_t(matrix_fragment_10);
            if (!condition_0) {
                matrix_fragment_14 = matrix_fragment_6 * uniform_index_word_4;

                matrix_fragment_7 = matrix_fragment_8;
                do {
                    matrix_fragment_4 = parameters[parameter_wait_optional_low_12];
                    matrix_fragment_5 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_9 = matrix_fragment_14 + matrix_fragment_7;

                    split_address_words(matrix_fragment_4, matrix_fragment_5,
                                        int64_t(int32_t(matrix_fragment_9)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(matrix_fragment_4, matrix_fragment_5));

                    matrix_fragment_9 = pipeline_read<CheckBounds>(
                        combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges,
                        error_counts);

                    matrix_fragment_7 = matrix_fragment_7 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_7) != uint32_t(matrix_fragment_10);

                    condition_0 = int32_t(matrix_fragment_9) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_9 = pipeline_read<CheckBounds>(
                                combine_words(matrix_fragment_4, matrix_fragment_5), memory_ranges,
                                error_counts);

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

    address_word_28 = parameters[parameter_tensor_height_8];
    index_word_29 = parameters[parameter_tensor_width_9];

    condition_4 = false;

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    condition_5 = false;

    matrix_fragment_14 = matrix_fragment_13 + 0x1u;

    matrix_fragment_4 = address_word_28 & 0xfffffffcu;

    matrix_fragment_5 = uint32_t(int32_t(address_word_28) >> 0x1fu);

    matrix_fragment_10 = index_word_29 & 0xfffffffcu;

    condition_0 = uint32_t(matrix_fragment_4) != uint32_t(0x4u);

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + address_word_28;

    matrix_fragment_4 = uint32_t(int32_t(index_word_29) >> 0x1fu);

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x2u >> 32) + index_word_29;

    matrix_fragment_15 = uint32_t(int32_t(matrix_fragment_4) >> 0x2u);

    if (condition_0) {
        matrix_fragment_6 = matrix_fragment_12 + 0x1u;

        condition_1 = int32_t(matrix_fragment_12) >= int32_t(matrix_fragment_5);
    }

    condition_3 = int32_t(matrix_fragment_13) >= int32_t(matrix_fragment_15);

    if (condition_0) {
        matrix_fragment_16 = matrix_fragment_12 * matrix_fragment_15 + matrix_fragment_15;

        condition_2 = int32_t(matrix_fragment_6) >= int32_t(matrix_fragment_5);

        condition_5 = int32_t(tile_word_0) < int32_t(0u - 0x3u) || condition_1;
    }

    condition_1 = true;

    if (condition_0) {
        matrix_fragment_17 = matrix_fragment_12 * matrix_fragment_15 + matrix_fragment_15;
    }

    condition_3 = int32_t(address_word_2) > int32_t(0u - 0x4u) && !condition_3;

    if (condition_0) {
        condition_4 = int32_t(tile_word_0) < int32_t(0u - 0x7u) || condition_2;

        condition_1 = !condition_5;
    }

    matrix_fragment_4 = uint32_t(condition_5) << 5;

    matrix_fragment_6 = uint32_t(condition_4) << 4;

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_5;

    condition_2 = true;

    condition_1 = condition_1 & (condition_4 | condition_3);

    matrix_fragment_9 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_fragment_6) != int32_t(0u);

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_1;

    if (condition_0) {
        condition_2 = !condition_1;
    }

    matrix_fragment_8 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_fragment_4) != int32_t(0u);

    if (condition_0) {
        matrix_fragment_4 = matrix_fragment_12 + 0x1u;
    }

    condition_2 = condition_2 & (condition_4 | condition_3);

    if (condition_0) {
        condition_4 = int32_t(matrix_fragment_12) >= int32_t(matrix_fragment_5);
    }

    condition_3 = false;

    if (condition_0) {
        condition_5 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_5);
    }

    matrix_fragment_6 = uint32_t(condition_2) << 2;

    if (condition_0) {
        condition_3 = int32_t(tile_word_0) < int32_t(0u - 0x3u) || condition_4;
    }

    condition_4 = false;

    if (condition_0) {
        condition_4 = int32_t(tile_word_0) < int32_t(0u - 0x7u) || condition_5;
    }

    condition_5 = int32_t(matrix_fragment_14) >= int32_t(matrix_fragment_15);

    condition_6 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_3;

    matrix_fragment_7 = uint32_t(condition_1) << 1;

    condition_1 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_4;

    condition_2 = int32_t(address_word_2) > int32_t(0u - 0x8u) && !condition_5;

    condition_5 = true;

    if (condition_0) {
        condition_5 = !condition_3;
    }

    matrix_fragment_11 = condition_3 ? matrix_fragment_14 : 0u;

    condition_3 = uint32_t(matrix_fragment_10) != uint32_t(0x4u);

    condition_5 = condition_5 & (condition_6 | condition_2);

    condition_6 = true;

    if (condition_0) {
        condition_6 = !condition_4;
    }

    address_word_2 = 0u;

    if (condition_0) {
        address_word_2 = matrix_fragment_12 * matrix_fragment_15;
    }

    condition_6 = condition_6 & (condition_1 | condition_2);

    condition_1 = int32_t(matrix_fragment_7) != int32_t(0u);

    condition_2 = int32_t(matrix_fragment_6) != int32_t(0u);

    matrix_fragment_21 = !condition_3 ? matrix_fragment_11 : matrix_fragment_14;

    if (condition_5) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];
    }

    if (!condition_3) {
        matrix_fragment_14 = condition_4 ? matrix_fragment_14 : 0u;
    }

    tile_word_0 = condition_1 ? matrix_fragment_13 : 0u;

    condition_1 = int32_t(matrix_fragment_8) != int32_t(0u);

    matrix_fragment_19 = !condition_3 ? tile_word_0 : matrix_fragment_13;

    if (!condition_3) {
        matrix_fragment_13 = condition_1 ? matrix_fragment_13 : 0u;
    }

    condition_1 = int32_t(matrix_fragment_9) != int32_t(0u);

    tile_word_0 = 0u;

    if (condition_0) {
        tile_word_0 = matrix_fragment_12 * matrix_fragment_15;
    }

    if (condition_5) {
        matrix_fragment_12 = matrix_fragment_21 + address_word_2;
    }

    if (condition_2) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];
    }

    address_word_2 = threadIdx.x & 31u;

    if (condition_2) {
        matrix_fragment_16 = matrix_fragment_13 + matrix_fragment_16;
    }

    if (condition_6) {
        matrix_fragment_14 = matrix_fragment_14 + matrix_fragment_17;
    }

    if (condition_5) {
        matrix_fragment_12 = matrix_fragment_12 * 0x8u + packed_half_3;
    }

    if (!condition_2) {
        matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_2) {
        matrix_fragment_16 = matrix_fragment_16 * 0x8u + packed_half_3;
    }

    if (condition_6) {
        matrix_fragment_10 = parameters[parameter_input_low_0];
        matrix_fragment_11 = parameters[parameter_input_high_1];
    }

    if (condition_1) {
        tile_word_0 = matrix_fragment_19 + tile_word_0;
    }

    if (condition_5) {
        matrix_fragment_15 = matrix_fragment_12 * 0x80u;
    }

    if (!condition_5) {
        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_6) {
        matrix_fragment_14 = matrix_fragment_14 * 0x8u + packed_half_3;
    }

    if (!condition_6) {
        matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_1) {
        tile_word_0 = tile_word_0 * 0x8u + packed_half_3;
    }

    if (!condition_5) {
        matrix_fragment_19 = permute_signed(matrix_fragment_19, matrix_fragment_19, 0x5410u);
    }

    if (condition_1) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_1) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_2) {
        matrix_fragment_23 = permute_signed(matrix_fragment_23, matrix_fragment_23, 0x5410u);
    }

    if (condition_1) {
        matrix_fragment_13 = tile_word_0 * 0x80u;
    }

    if (!condition_1) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_2) {
        matrix_fragment_17 = matrix_fragment_16 * 0x80u;
    }

    if (!condition_6) {
        matrix_fragment_27 = permute_signed(matrix_fragment_27, matrix_fragment_27, 0x5410u);
    }

    if (!condition_5) {
        matrix_fragment_16 = matrix_fragment_19;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_1) {
        matrix_fragment_12 = matrix_fragment_15;
    }

    if (!condition_5) {
        matrix_fragment_17 = matrix_fragment_19;

        matrix_fragment_18 = matrix_fragment_19;
    }

    if (!condition_2) {
        matrix_fragment_20 = matrix_fragment_23;

        matrix_fragment_21 = matrix_fragment_23;

        matrix_fragment_22 = matrix_fragment_23;
    }

    if (!condition_6) {
        matrix_fragment_24 = matrix_fragment_27;

        matrix_fragment_25 = matrix_fragment_27;

        matrix_fragment_26 = matrix_fragment_27;
    }

    if (condition_5) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    __syncthreads();

    if (condition_1) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_6) {
        matrix_fragment_13 = matrix_fragment_14 * 0x80u;
    }

    if (!condition_1) {
        matrix_fragment_14 = matrix_fragment_15;
    }

    if (condition_1) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_6) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (!condition_1) {
        matrix_fragment_13 = matrix_fragment_15;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (condition_6) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }

    if (condition_6) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_10, matrix_fragment_11), matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);
    }

    uniform_index_word_5 = 0u;

    uniform_index_word_4 = 0x400u;

    uniform_parameter_word_6 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_7 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_86 = 0;
    matrix_fragment_87 = 0;

    matrix_fragment_94 = 0;
    matrix_fragment_95 = 0;

    matrix_fragment_96 = 0;
    matrix_fragment_97 = 0;

    matrix_accumulator_76 = 0;
    matrix_accumulator_77 = 0;

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    packed_half_78 = 0;
    packed_half_79 = 0;

    matrix_accumulator_82 = 0;
    packed_half_83 = 0;

    packed_half_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_80 = 0;
    matrix_accumulator_81 = 0;

    matrix_fragment_84 = 0;
    matrix_fragment_85 = 0;

    matrix_fragment_90 = 0;
    matrix_fragment_91 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_fragment_88 = 0;
    matrix_fragment_89 = 0;

    matrix_fragment_92 = 0;
    matrix_fragment_93 = 0;

    matrix_fragment_98 = 0;
    matrix_fragment_99 = 0;

    split_address_words(packed_half_106, matrix_accumulator_107,
                        combine_words(uniform_parameter_word_6, uniform_parameter_word_7));

    uniform_index_word_5 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

    uniform_index_word_4 = 0u;

    matrix_accumulator_31 = (packed_half_3 << 0x9u) + uniform_index_word_5;

    tile_word_0 = address_word_2 * 0x10u + matrix_accumulator_31;

    packed_half_30 = 0;
    matrix_accumulator_31 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x1000u, matrix_fragment_16,
                                         matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x2000u, matrix_fragment_20,
                                         matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x3000u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    __syncthreads();
    do {
        split_address_words(matrix_fragment_118, matrix_fragment_119,
                            uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x8000u)) +
                                combine_words(packed_half_106, matrix_accumulator_107));

        split_address_words(matrix_fragment_118, matrix_fragment_119,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_118, matrix_fragment_119));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_118, matrix_fragment_119),
                                            matrix_fragment_4, matrix_fragment_5, matrix_fragment_6,
                                            matrix_fragment_7, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x1000ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x1200ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x2200ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x2000ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x3200ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x3000ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x4000ull, matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x4200ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        matrix_fragment_112 = (address_word_2 << 0x4u) + uniform_index_word_5;

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x5000ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_112 + 0x1000u,
                                            matrix_accumulator_124, matrix_accumulator_125,
                                            matrix_accumulator_126, matrix_accumulator_127,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2000u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x5200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_112,
                                            matrix_accumulator_140, matrix_accumulator_141,
                                            matrix_accumulator_142, matrix_accumulator_143,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3000u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x200u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1200u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3200u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133,
                                 matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_4,
                                 matrix_fragment_5, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_6,
                                 matrix_fragment_7, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131,
                                 matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                                 matrix_fragment_9, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_10,
                                 matrix_fragment_11, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_4, matrix_fragment_5, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_6, matrix_fragment_7, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_8, matrix_fragment_9, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_10, matrix_fragment_11, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2200u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137,
                                 matrix_accumulator_140, matrix_accumulator_141,
                                 matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_4,
                                 matrix_fragment_5, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_accumulator_140, matrix_accumulator_141,
                                 matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_8,
                                 matrix_fragment_9, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139,
                                 matrix_accumulator_140, matrix_accumulator_141,
                                 matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_6,
                                 matrix_fragment_7, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143,
                                 matrix_accumulator_140, matrix_accumulator_141,
                                 matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_10,
                                 matrix_fragment_11, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x6000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_8, matrix_fragment_9, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_10, matrix_fragment_11, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x6200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1400u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_108,
                                 matrix_accumulator_109);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_112 + 0x400u,
                                            matrix_accumulator_108, matrix_accumulator_109,
                                            matrix_accumulator_110, matrix_accumulator_111,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_144,
                                 matrix_accumulator_145);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143,
                                 matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_52, matrix_fragment_53);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x7000ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        multiply_accumulate_half(
            matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_108,
            matrix_accumulator_109, matrix_accumulator_110, matrix_accumulator_111,
            matrix_fragment_44, matrix_fragment_45, matrix_accumulator_136, matrix_accumulator_137);

        multiply_accumulate_half(
            matrix_accumulator_138, matrix_accumulator_139, matrix_accumulator_108,
            matrix_accumulator_109, matrix_accumulator_110, matrix_accumulator_111,
            matrix_fragment_46, matrix_fragment_47, matrix_accumulator_138, matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3400u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x600u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_140,
                                 matrix_accumulator_141);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2400u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1600u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_accumulator_108, matrix_accumulator_109,
                                 matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_50,
                                 matrix_fragment_51, matrix_fragment_62, matrix_fragment_63);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_60,
                                 matrix_fragment_61);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_118, matrix_fragment_119) + 0x7200ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2600u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3600u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1800u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_140,
                                 matrix_accumulator_141);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x800u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2800u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_140,
                                 matrix_accumulator_141);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3800u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1a00u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0xa00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2a00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_140,
                                 matrix_accumulator_141);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3a00u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_24, matrix_fragment_25, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_26, matrix_fragment_27, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2c00u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_114,
                                 matrix_accumulator_115);

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x1000u)) +
                                combine_words(packed_half_106, matrix_accumulator_107));

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_138,
                                 matrix_accumulator_139);

        split_address_words(matrix_fragment_48, matrix_fragment_49,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_48, matrix_fragment_49) + 0x40000ull, matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_140,
                                 matrix_accumulator_141);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_48, matrix_fragment_49) + 0x40200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1c00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_138,
                                 matrix_accumulator_139);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0xc00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0xe00u, matrix_fragment_32, matrix_fragment_33,
            matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x3c00u, matrix_fragment_16, matrix_fragment_17,
            matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x1e00u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_6, matrix_fragment_7, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_8, matrix_fragment_9, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_112 + 0x2e00u, matrix_fragment_24, matrix_fragment_25,
            matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_42,
                                 matrix_fragment_43);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_4, matrix_fragment_5, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_6, matrix_fragment_7, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

        matrix_fragment_38 = 0x2b28u;

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_50,
                                 matrix_fragment_51);

        matrix_fragment_39 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

        matrix_fragment_57 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_39),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_112 + 0x3e00u,
                                            matrix_fragment_112, matrix_fragment_113,
                                            matrix_accumulator_114, matrix_accumulator_115,
                                            shared_bytes, memory_ranges, error_counts);

        matrix_fragment_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_57),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57),
            select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

        matrix_fragment_57 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_24,
                                 matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_128,
                                 matrix_accumulator_129);

        matrix_fragment_24 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

        matrix_fragment_25 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_26 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_26),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_25 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

        matrix_fragment_24 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_24 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26),
            select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0x44004400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_24),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0x44004400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_24),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_fragment_112, matrix_fragment_113, matrix_accumulator_114,
                                 matrix_accumulator_115, matrix_fragment_52, matrix_fragment_53,
                                 matrix_accumulator_116, matrix_accumulator_117);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

        matrix_fragment_18 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

        matrix_fragment_25 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

        matrix_fragment_53 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_112,
                                 matrix_fragment_113, matrix_accumulator_114,
                                 matrix_accumulator_115, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_6, matrix_fragment_7);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_26 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

        matrix_fragment_7 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_4 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_7),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_4 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_112,
                                 matrix_fragment_113, matrix_accumulator_114,
                                 matrix_accumulator_115, matrix_fragment_60, matrix_fragment_61,
                                 matrix_fragment_8, matrix_fragment_9);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_57),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57),
            select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

        matrix_fragment_57 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        matrix_fragment_6 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_112,
                                 matrix_fragment_113, matrix_accumulator_114,
                                 matrix_accumulator_115, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_10, matrix_fragment_11);

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_19 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_53),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_57),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_26),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_6 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_6), 0x3b283b28u);

        matrix_fragment_9 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_16 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_4 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57),
            select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

        matrix_fragment_27 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26),
            select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

        matrix_fragment_26 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_128), 0x44004400u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_6 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

        matrix_fragment_17 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_9 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

        matrix_fragment_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_5 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_4 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

        matrix_fragment_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_129), 0x44004400u);

        matrix_fragment_35 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_7 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

        matrix_fragment_8 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_9),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_6 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_4),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_26 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_9),
            select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

        matrix_fragment_7 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_6),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_27 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_9 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_8),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_34 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_4),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

        matrix_fragment_5 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_6),
            select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_8),
            select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

        matrix_fragment_18 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

        matrix_fragment_32 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                          select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_11), 0xc400c400u);

        matrix_fragment_25 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

        matrix_fragment_39 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_33),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_18 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_19),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_25),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_39),
            select_half_operand<0, false, true>(matrix_fragment_38), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33),
            select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39),
            select_half_operand<2, false, false>(matrix_fragment_38), 0x3b283b28u);

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_24 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19),
            select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_25),
            select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

        matrix_fragment_37 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_33 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_26 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_26));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_34 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_36));

        matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_24 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                select_half_operand<2, false, false>(matrix_fragment_24));

        matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_32 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                          select_half_operand<2, false, false>(matrix_fragment_19));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                          select_half_operand<2, false, false>(matrix_fragment_27));

        matrix_fragment_19 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                          select_half_operand<2, false, false>(matrix_fragment_25));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                          select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                          select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_20, matrix_fragment_21, matrix_fragment_86,
                                 matrix_fragment_87);

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_6 = uniform_index_word_4 + 0x20u;

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(packed_half_104, matrix_accumulator_105, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_20, matrix_fragment_21, packed_half_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_20, matrix_fragment_21, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_94,
                                 matrix_fragment_95);

        multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_22, matrix_fragment_23, packed_half_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_22, matrix_fragment_23, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_22, matrix_fragment_23, matrix_fragment_88,
                                 matrix_fragment_89);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_76,
                                 matrix_accumulator_77);

        multiply_accumulate_half(packed_half_78, packed_half_79, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_12, matrix_fragment_13, packed_half_78,
                                 packed_half_79);

        multiply_accumulate_half(matrix_accumulator_82, packed_half_83, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_82,
                                 packed_half_83);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_84,
                                 matrix_fragment_85);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_90,
                                 matrix_fragment_91);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_92,
                                 matrix_fragment_93);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_98,
                                 matrix_fragment_99);

        split_address_words(packed_half_106, matrix_accumulator_107,
                            combine_words(packed_half_106, matrix_accumulator_107) + 0x400u);

        uniform_index_word_4 = uniform_parameter_word_6;
    } while (!uniform_condition_0);
    address_word_162 = parameters[parameter_trained_weights_low_4];
    address_word_163 = parameters[parameter_trained_weights_high_5];

    index_word_165 = address_word_2;

    matrix_fragment_10 = packed_half_3 * 0x20u;

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_9 = index_word_165 & 0x7ffffffcu;

    index_word_164 = uint32_t(uint32_t(matrix_fragment_10) >> 0x1u);

    index_word_161 = index_word_165 & 0xfffffffcu;

    matrix_fragment_11 = 0u - matrix_fragment_9 + address_word_2;

    matrix_fragment_6 = index_word_164 + 0x8u;

    address_word_160 = index_word_164 + 0x4u;

    matrix_fragment_8 = index_word_164 + 0xcu;

    matrix_fragment_11 = matrix_fragment_11 * 0x2u + matrix_fragment_10;

    matrix_fragment_7 = matrix_fragment_6 + address_word_2 + (0u - index_word_161);

    address_word_160 = address_word_160 + address_word_2 + (0u - index_word_161);

    matrix_fragment_13 = matrix_fragment_8 + address_word_2 + (0u - index_word_161);

    matrix_fragment_11 = uint32_t(int32_t(matrix_fragment_11) >> 0x1u);

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        uint64_t(uint32_t(matrix_fragment_7)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(address_word_160)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_49 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_13)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_47 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(packed_half_152, address_word_153,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_48 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x58010ull, memory_ranges,
        error_counts);

    matrix_fragment_46 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_152, address_word_153) + 0x58010ull, memory_ranges, error_counts);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x1000u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_78),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_79),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x2000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                    select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                      select_half_operand<2, false, false>(matrix_fragment_16));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x3000u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    __syncthreads();

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                      select_half_operand<2, false, false>(0u));

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                      select_half_operand<2, false, false>(0u));

    packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                      select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                      select_half_operand<2, false, false>(packed_half_30));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x1000u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_0 + 0x2000u, packed_half_104, matrix_accumulator_105,
        packed_half_106, matrix_accumulator_107, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_0 + 0x3000u, matrix_accumulator_100, matrix_accumulator_101,
        matrix_accumulator_102, matrix_accumulator_103, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_42 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_43 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_162, address_word_163));

    packed_half_30 = unpack_fp8_bits(matrix_fragment_20);

    matrix_accumulator_31 = unpack_fp8_bits(matrix_fragment_21);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_22);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    split_address_words(matrix_fragment_18, matrix_fragment_19,
                        combine_words(matrix_fragment_18, matrix_fragment_19) + 0x48200u);

    uniform_index_word_4 = 0u;

    uniform_parameter_word_6 = uniform_index_word_5;

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    packed_half_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                         select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5) - 0x200ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                            matrix_fragment_56, matrix_fragment_57,
                                            matrix_fragment_58, matrix_fragment_59, memory_ranges,
                                            error_counts);

        matrix_fragment_6 = (address_word_2 << 0x4u) + uniform_parameter_word_6;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            combine_words(matrix_fragment_18, matrix_fragment_19) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_6 = uniform_parameter_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x1000u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x2000u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_6 + 0x3000u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_52, matrix_fragment_53, packed_half_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_52, matrix_fragment_53, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_52, matrix_fragment_53, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_fragment_44,
                                 matrix_fragment_45);
    } while (!uniform_condition_0);
    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_60, matrix_fragment_61,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                        combine_words(matrix_fragment_60, matrix_fragment_61) + 0x58c20u);

    packed_half_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_102 = 0;
    matrix_accumulator_103 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = 0;
    matrix_fragment_21 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_112 = 0;
    matrix_fragment_113 = 0;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_30 = 0;
    matrix_accumulator_31 = 0;

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_118 = 0;
    matrix_fragment_119 = 0;

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x2000u, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_54 = 0;
    matrix_fragment_55 = 0;

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_16 = 0;
    matrix_fragment_17 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x1000u, matrix_fragment_48,
                                         matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_fragment_56 = 0;
    matrix_fragment_57 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x3000u, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_fragment_58 = 0;
    matrix_fragment_59 = 0;

    matrix_accumulator_100 = 0;
    matrix_accumulator_101 = 0;

    matrix_accumulator_140 = 0;
    matrix_accumulator_141 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_138 = 0;
    matrix_accumulator_139 = 0;

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_accumulator_132 = 0;
    matrix_accumulator_133 = 0;

    uniform_index_word_4 = 0u;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    uniform_parameter_word_6 = uniform_index_word_5;

    __syncthreads();
    do {
        split_address_words(matrix_accumulator_144, matrix_accumulator_145,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_142, matrix_accumulator_143));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_144, matrix_accumulator_145) - 0xa00ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_144, matrix_accumulator_145) - 0x800ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_144, matrix_accumulator_145) - 0x600ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_144, matrix_accumulator_145) - 0x400ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_144, matrix_accumulator_145) - 0x200ull,
            matrix_accumulator_76, matrix_accumulator_77, packed_half_78, packed_half_79,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_144, matrix_accumulator_145), matrix_accumulator_80,
            matrix_accumulator_81, matrix_accumulator_82, packed_half_83, memory_ranges,
            error_counts);

        packed_half_3 = (address_word_2 << 0x4u) + uniform_parameter_word_6;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                            combine_words(matrix_accumulator_142, matrix_accumulator_143) +
                                0x6000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_3, matrix_fragment_84, matrix_fragment_85, matrix_fragment_86,
            matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_6 = uniform_parameter_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_3 + 0x1000u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_3 + 0x2000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_3 + 0x3000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, packed_half_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_fragment_118, matrix_fragment_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_118,
                                 matrix_fragment_119);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_102,
                                 matrix_accumulator_103);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_112, matrix_fragment_113, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_112,
                                 matrix_fragment_113);

        multiply_accumulate_half(packed_half_104, matrix_accumulator_105, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, packed_half_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(packed_half_106, matrix_accumulator_107, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, packed_half_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_76, matrix_accumulator_77,
                                 matrix_accumulator_100, matrix_accumulator_101);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 packed_half_78, packed_half_79, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 packed_half_78, packed_half_79, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 packed_half_78, packed_half_79, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 packed_half_78, packed_half_79, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_82, packed_half_83, matrix_fragment_54,
                                 matrix_fragment_55);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_82, packed_half_83, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_accumulator_82, packed_half_83, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_accumulator_82, packed_half_83, matrix_fragment_26,
                                 matrix_fragment_27);
    } while (!uniform_condition_0);
    matrix_fragment_63 = threadIdx.y;

    packed_half_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    split_address_words(matrix_fragment_64, matrix_fragment_65,
                        uint64_t(uint32_t(matrix_fragment_63)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_64 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_64, matrix_fragment_65) + 0x98220ull, memory_ranges,
        error_counts);

    matrix_accumulator_143 = matrix_fragment_63 * 0x800u;

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(packed_half_3));

    __syncthreads();

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                        uint64_t(uint32_t(matrix_accumulator_143)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                        int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_142, matrix_accumulator_143));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                       select_half_operand<2, false, false>(matrix_fragment_66));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88420ull,
        matrix_fragment_92, matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
        memory_ranges, error_counts);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                       select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88620ull,
        matrix_fragment_96, matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
        memory_ranges, error_counts);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88220ull,
        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
        memory_ranges, error_counts);

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88820ull,
        matrix_accumulator_148, matrix_accumulator_149, matrix_accumulator_150,
        matrix_accumulator_151, memory_ranges, error_counts);

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                       select_half_operand<2, false, false>(matrix_fragment_69));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88a20ull,
        matrix_accumulator_144, matrix_accumulator_145, matrix_accumulator_146,
        matrix_accumulator_147, memory_ranges, error_counts);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_accumulator_141));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_80), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_83 = result_word;
    }

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    packed_half_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                            select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(matrix_fragment_32));

    {
        const uint32_t bits = address_word_2;
        condition_0 = bool(bits & 1u);
        condition_1 = bool(bits & 2u);
        condition_3 = bool(bits & 8u);
        condition_4 = bool(bits & 16u);
    }

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_45 = transpose_half_tile(matrix_fragment_45);

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_118),
                                      select_half_operand<2, false, false>(matrix_fragment_118),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_59 = transpose_half_tile(matrix_fragment_59);

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                   select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_58 = transpose_half_tile(matrix_fragment_58);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_accumulator_101 = transpose_half_tile(matrix_accumulator_101);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_119),
                                      select_half_operand<2, false, false>(matrix_fragment_119),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    packed_half_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                      select_half_operand<2, false, false>(packed_half_104),
                                      select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_57 = transpose_half_tile(matrix_fragment_57);

    matrix_accumulator_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_77));

    packed_half_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                      select_half_operand<2, false, false>(matrix_fragment_112),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_100 = transpose_half_tile(matrix_accumulator_100);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_56 = transpose_half_tile(matrix_fragment_56);

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_43 = transpose_half_tile(matrix_fragment_43);

    matrix_fragment_70 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_78),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_3), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_78 = result_word;
    }

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                        select_half_operand<2, false, false>(packed_half_83));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_83 = result_word;
    }

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(packed_half_79),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_79 = result_word;
    }

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                      select_half_operand<2, false, false>(matrix_fragment_113),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_42 = transpose_half_tile(matrix_fragment_42);

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_75 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                             select_half_operand<2, false, false>(packed_half_30));

    packed_half_152 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_152, address_word_153) + 0xa8240ull, memory_ranges, error_counts);

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_81), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    packed_half_154 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_82 = result_word;
    }

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                   select_half_operand<2, false, false>(packed_half_78));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(packed_half_83));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_78 = result_word;
    }

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(packed_half_79));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_77 = result_word;
    }

    packed_half_79 = permute_signed(matrix_accumulator_76, matrix_accumulator_76, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_80 = result_word;
    }

    packed_half_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                                    select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_fragment_86));

    packed_half_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_79), 0x4100410u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_69 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_82 = half_to_float_bits((packed_half_79 & 65535u) * 65537u);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_81), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_76 = result_word;
    }

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_88 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_accumulator_105),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    packed_half_83 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                    select_half_operand<2, false, false>(packed_half_78));

    packed_half_78 = rsqrt_bits(matrix_accumulator_82);

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_71 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    matrix_accumulator_82 = permute_signed(matrix_fragment_75, matrix_fragment_75, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_77 = result_word;
    }

    matrix_fragment_67 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88e20ull,
        matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
        memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_80 = result_word;
    }

    matrix_accumulator_82 = permute_signed(packed_half_83, packed_half_83, 0x5432u);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_76 = result_word;
    }

    packed_half_83 = add_half_pairs(select_half_operand<2, false, false>(packed_half_83),
                                    select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_65 = result_word;
    }

    packed_half_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_83), 0x4100410u);

    matrix_fragment_66 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x4100410u);

    matrix_accumulator_82 = half_to_float_bits((packed_half_83 & 65535u) * 65537u);

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_75 = permute_signed(matrix_fragment_74, matrix_fragment_74, 0x5432u);

    packed_half_83 = half_to_float_bits((packed_half_83 >> 16) * 65537u);

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                   select_half_operand<2, false, false>(matrix_accumulator_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_77 = result_word;
    }

    matrix_fragment_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_80 = result_word;
    }

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_75 = rsqrt_bits(packed_half_83);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                     select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_85 = half_to_float_bits((matrix_fragment_67 & 65535u) * 65537u);

    packed_half_155 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x4100410u);

    matrix_fragment_68 = rsqrt_bits(matrix_accumulator_82);

    matrix_fragment_67 = half_to_float_bits((matrix_fragment_67 >> 16) * 65537u);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                     select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_87 = half_to_float_bits((matrix_fragment_74 >> 16) * 65537u);

    matrix_fragment_69 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_76 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_fragment_84 = rsqrt_bits(matrix_fragment_67);

    matrix_accumulator_82 = half_to_float_bits((matrix_fragment_74 & 65535u) * 65537u);

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_67 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5432u);

    matrix_accumulator_82 = rsqrt_bits(matrix_accumulator_82);

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_87 = rsqrt_bits(matrix_fragment_87);

    packed_half_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    matrix_fragment_75 = pack_half(matrix_fragment_75, matrix_fragment_68);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x4100410u);

    matrix_fragment_86 = permute_signed(matrix_fragment_69, matrix_fragment_69, 0x5432u);

    matrix_fragment_71 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_68 = half_to_float_bits((packed_half_83 & 65535u) * 65537u);

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    packed_half_83 = half_to_float_bits((packed_half_83 >> 16) * 65537u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_138 = half_to_float_bits((matrix_fragment_70 & 65535u) * 65537u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_86 = half_to_float_bits((matrix_fragment_70 >> 16) * 65537u);

    matrix_fragment_68 = rsqrt_bits(matrix_fragment_68);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x4100410u);

    packed_half_158 = pack_half(matrix_fragment_87, matrix_accumulator_82);

    packed_half_83 = rsqrt_bits(packed_half_83);

    matrix_accumulator_82 = permute_signed(matrix_fragment_71, matrix_fragment_71, 0x5432u);

    matrix_accumulator_138 = rsqrt_bits(matrix_accumulator_138);

    matrix_fragment_69 = rsqrt_bits(matrix_fragment_86);

    matrix_fragment_71 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x4100410u);

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_134 = half_to_float_bits((matrix_fragment_70 & 65535u) * 65537u);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_82 = half_to_float_bits((matrix_fragment_70 >> 16) * 65537u);

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_86 = half_to_float_bits((matrix_fragment_71 & 65535u) * 65537u);

    packed_half_156 = pack_half(packed_half_83, matrix_fragment_68);

    matrix_fragment_87 = half_to_float_bits((matrix_fragment_71 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x88c20ull,
        matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
        memory_ranges, error_counts);

    matrix_accumulator_138 = pack_half(matrix_fragment_69, matrix_accumulator_138);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89020ull,
        matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
        memory_ranges, error_counts);

    packed_half_79 = half_to_float_bits((packed_half_79 >> 16) * 65537u);

    packed_half_79 = rsqrt_bits(packed_half_79);

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_134);

    matrix_accumulator_82 = rsqrt_bits(matrix_accumulator_82);

    packed_half_78 = pack_half(packed_half_79, packed_half_78);

    matrix_fragment_85 = rsqrt_bits(matrix_fragment_85);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_78));

    packed_half_78 = permute_signed(matrix_fragment_65, matrix_fragment_65, 0x5432u);

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_accumulator_140 = pack_half(matrix_accumulator_82, matrix_accumulator_134);

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(packed_half_78));

    packed_half_158 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_84 = pack_half(matrix_fragment_84, matrix_fragment_85);

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_118),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(packed_half_156));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x4100410u);

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_110 = pack_half(0u, matrix_fragment_64);

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    packed_half_159 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_119),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_114 = half_to_float_bits((matrix_fragment_85 & 65535u) * 65537u);

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_116 = half_to_float_bits((matrix_fragment_85 >> 16) * 65537u);

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = rsqrt_bits(matrix_accumulator_114);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_127 = rsqrt_bits(matrix_accumulator_116);

    packed_half_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_76 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_86 = rsqrt_bits(matrix_fragment_86);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_118 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_87 = rsqrt_bits(matrix_fragment_87);

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_119 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                      select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89220ull,
        matrix_accumulator_76, matrix_accumulator_77, packed_half_78, packed_half_79, memory_ranges,
        error_counts);

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_127 = pack_half(matrix_accumulator_127, matrix_accumulator_114);

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                       select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_116 = permute_signed(matrix_accumulator_81, matrix_accumulator_81, 0x5432u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(0u));

    packed_half_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_118, matrix_fragment_119, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_112 = pack_half(matrix_fragment_87, matrix_fragment_86);

    matrix_accumulator_132 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_116));

    packed_half_156 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                    select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89420ull,
        matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
        memory_ranges, error_counts);

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_159),
                      select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_112));

    matrix_accumulator_122 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                            select_half_operand<2, false, false>(matrix_fragment_112));

    matrix_accumulator_81 = permute_signed(matrix_accumulator_80, matrix_accumulator_80, 0x5432u);

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_fragment_112));

    matrix_fragment_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_83),
                                        select_half_operand<2, false, false>(packed_half_156));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_112));

    matrix_fragment_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_80 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_94,
                             matrix_fragment_95);

    packed_half_158 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_158),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_125 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_140 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_94 = permute_signed(packed_half_3, packed_half_3, 0x5432u);

    matrix_accumulator_122 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                      select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(packed_half_158));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                   select_half_operand<2, false, false>(matrix_fragment_94));

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_112, matrix_fragment_113, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_accumulator_124 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                      select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_125 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80), 0x4100410u);

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89820ull,
        matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_82, packed_half_83,
        memory_ranges, error_counts);

    matrix_fragment_92 = half_to_float_bits((matrix_fragment_93 & 65535u) * 65537u);

    matrix_fragment_95 = half_to_float_bits((matrix_fragment_93 >> 16) * 65537u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_138 = half_to_float_bits((matrix_accumulator_132 & 65535u) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89620ull,
        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
        memory_ranges, error_counts);

    matrix_fragment_94 = half_to_float_bits((matrix_accumulator_132 >> 16) * 65537u);

    matrix_fragment_92 = rsqrt_bits(matrix_fragment_92);

    matrix_fragment_95 = rsqrt_bits(matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_fragment_96 = half_to_float_bits((matrix_fragment_93 & 65535u) * 65537u);

    matrix_fragment_97 = half_to_float_bits((matrix_fragment_93 >> 16) * 65537u);

    matrix_accumulator_138 = rsqrt_bits(matrix_accumulator_138);

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                      select_half_operand<2, false, false>(0u));

    packed_half_3 = rsqrt_bits(matrix_fragment_94);

    matrix_fragment_96 = rsqrt_bits(matrix_fragment_96);

    matrix_fragment_97 = rsqrt_bits(matrix_fragment_97);

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                      select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_111 = pack_half(matrix_fragment_95, matrix_fragment_92);

    matrix_accumulator_136 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_138 = pack_half(packed_half_3, matrix_accumulator_138);

    matrix_accumulator_136 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_half(matrix_fragment_97, matrix_fragment_96);

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_3));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89a20ull,
        matrix_fragment_92, matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
        memory_ranges, error_counts);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                      select_half_operand<2, false, false>(matrix_accumulator_137));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89c20ull,
        matrix_fragment_96, matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
        memory_ranges, error_counts);

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    multiply_accumulate_half(matrix_accumulator_148, matrix_accumulator_149, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_148,
                             matrix_accumulator_149);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    packed_half_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_150,
                             matrix_accumulator_151);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(matrix_fragment_64));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x89e20ull,
        matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x8a020ull,
        matrix_accumulator_140, matrix_accumulator_141, matrix_accumulator_142,
        matrix_accumulator_143, memory_ranges, error_counts);

    packed_half_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    packed_half_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(packed_half_104));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_154), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_3 = result_word;
    }

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_52 = result_word;
    }

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                   select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_46 = permute_signed(packed_half_3, packed_half_3, 0x5432u);

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                   select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x4100410u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_46 = half_to_float_bits((packed_half_3 & 65535u) * 65537u);

    matrix_fragment_53 = half_to_float_bits((packed_half_3 >> 16) * 65537u);

    matrix_fragment_47 = permute_signed(matrix_fragment_52, matrix_fragment_52, 0x5432u);

    matrix_fragment_46 = rsqrt_bits(matrix_fragment_46);

    matrix_fragment_53 = rsqrt_bits(matrix_fragment_53);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x4100410u);

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    packed_half_104 = half_to_float_bits((matrix_fragment_52 & 65535u) * 65537u);

    matrix_accumulator_105 = half_to_float_bits((matrix_fragment_52 >> 16) * 65537u);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    packed_half_106 = pack_half(matrix_fragment_53, matrix_fragment_46);

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_53 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_52 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    packed_half_104 = rsqrt_bits(packed_half_104);

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                        select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(packed_half_106));

    packed_half_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                         select_half_operand<2, false, false>(packed_half_106));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                   select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_105 = rsqrt_bits(matrix_accumulator_105);

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_118, matrix_fragment_119, matrix_accumulator_144,
                             matrix_accumulator_145);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_146,
                             matrix_accumulator_147);

    packed_half_104 = pack_half(matrix_accumulator_105, packed_half_104);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_112, matrix_fragment_113, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_3), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_30 = result_word;
    }

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                   select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_30 = result_word;
    }

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_31 = result_word;
    }

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                    select_half_operand<2, false, false>(packed_half_30));

    packed_half_3 = permute_signed(packed_half_30, packed_half_30, 0x5432u);

    matrix_accumulator_31 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                       select_half_operand<2, false, false>(matrix_accumulator_31));

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                    select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_36 = permute_signed(matrix_accumulator_31, matrix_accumulator_31, 0x5432u);

    packed_half_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x4100410u);

    matrix_accumulator_31 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                       select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_3 = half_to_float_bits((packed_half_30 & 65535u) * 65537u);

    matrix_accumulator_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31), 0x4100410u);

    packed_half_30 = half_to_float_bits((packed_half_30 >> 16) * 65537u);

    matrix_fragment_36 = half_to_float_bits((matrix_accumulator_31 & 65535u) * 65537u);

    matrix_accumulator_31 = half_to_float_bits((matrix_accumulator_31 >> 16) * 65537u);

    packed_half_3 = rsqrt_bits(packed_half_3);

    packed_half_30 = rsqrt_bits(packed_half_30);

    matrix_fragment_36 = rsqrt_bits(matrix_fragment_36);

    matrix_accumulator_31 = rsqrt_bits(matrix_accumulator_31);

    matrix_fragment_37 = pack_half(packed_half_30, packed_half_3);

    packed_half_30 = 0x29c0u;

    matrix_accumulator_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_120),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_36 = pack_half(matrix_accumulator_31, matrix_fragment_36);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_120 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_120), 0x3c203c20u);

    matrix_accumulator_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_121),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_114 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3c203c20u);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_116 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3c203c20u);

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_128 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3c203c20u);

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_120 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_120), 0x3e473e47u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_114 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3e473e47u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_116 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3e473e47u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_128 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3e473e47u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_121 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_121), 0x3c203c20u);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_115 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3c203c20u);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_117 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3c203c20u);

    matrix_accumulator_132 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_129 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3c203c20u);

    matrix_accumulator_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_121 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_121), 0x3e473e47u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_115 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3e473e47u);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_117 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3e473e47u);

    matrix_accumulator_148 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_148),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_129 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3e473e47u);

    matrix_accumulator_150 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_150),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_105 = (matrix_accumulator_120 << 0x5u) + 0x7ff88000u;

    packed_half_104 = (matrix_accumulator_114 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 = (matrix_accumulator_116 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 = (matrix_accumulator_128 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x3c203c20u);

    matrix_accumulator_134 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_118, matrix_fragment_119, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                   select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_46 = (matrix_accumulator_121 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = (matrix_accumulator_117 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x3e473e47u);

    multiply_accumulate_half(packed_half_78, packed_half_79, matrix_fragment_32, matrix_fragment_33,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_122,
                             matrix_accumulator_123, packed_half_78, packed_half_79);

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3e473e47u);

    matrix_accumulator_148 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_148), 0x3c203c20u);

    matrix_accumulator_150 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_150), 0x3c203c20u);

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_133 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_148 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_148), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_112, matrix_fragment_113, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_accumulator_150 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_150), 0x3e473e47u);

    matrix_accumulator_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_17 = (matrix_accumulator_132 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_149 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_149),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_accumulator_151 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_151),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_133 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133), 0x3c203c20u);

    matrix_accumulator_135 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_82, packed_half_83, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_82,
                             packed_half_83);

    matrix_fragment_35 = (matrix_accumulator_115 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (matrix_accumulator_129 << 0x5u) + 0x7ff88000u;

    matrix_fragment_32 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                   select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_3 = (matrix_accumulator_148 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = (matrix_accumulator_150 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_accumulator_133 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133), 0x3e473e47u);

    matrix_accumulator_144 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_144),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3e473e47u);

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_149 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_149), 0x3c203c20u);

    matrix_accumulator_145 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_145),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_151 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_151), 0x3c203c20u);

    matrix_accumulator_146 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_146),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_147 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_147),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_149 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_149), 0x3e473e47u);

    matrix_accumulator_151 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_151), 0x3e473e47u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_18 = (matrix_accumulator_133 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_31 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_144 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_144), 0x3c203c20u);

    matrix_accumulator_145 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_145), 0x3c203c20u);

    matrix_accumulator_146 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_146), 0x3c203c20u);

    matrix_accumulator_147 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_147), 0x3c203c20u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_50 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_19 = (matrix_accumulator_149 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_34 = (matrix_accumulator_151 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_144 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_144), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_145 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_145), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_146 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_146), 0x3e473e47u);

    matrix_accumulator_147 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_147), 0x3e473e47u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_73 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_72 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                       select_half_operand<2, false, false>(matrix_fragment_34));

    packed_half_106 = (matrix_accumulator_144 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_108 = (matrix_accumulator_145 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_109 = (matrix_accumulator_146 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_107 = (matrix_accumulator_147 << 0x5u) + 0x7ff88000u;

    matrix_fragment_75 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_74 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                       select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_118, matrix_fragment_119, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_fragment_47 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                       select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_50 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                       select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_52 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_112, matrix_fragment_113, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_fragment_70 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_112 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                         select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_47 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_113 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                         select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_50 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_51 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_49 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_68 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_64,
                             matrix_fragment_65);

    packed_half_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_78),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    packed_half_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_78), 0x3c203c20u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    packed_half_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_79),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_140,
                             matrix_accumulator_141);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_89 = condition_0 ? matrix_accumulator_114 : matrix_fragment_69;

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_114 = condition_0 ? matrix_fragment_69 : matrix_accumulator_114;

    packed_half_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_78), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_142,
                             matrix_accumulator_143);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_accumulator_77 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3c203c20u);

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    packed_half_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_79), 0x3c203c20u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    packed_half_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_79), 0x3e473e47u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_fragment_86 = (packed_half_78 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_85 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    packed_half_78 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    packed_half_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_83),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_88 = condition_0 ? matrix_accumulator_111 : matrix_accumulator_110;

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_77 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_76 = (packed_half_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_71 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_84 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    packed_half_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                    select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_60 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(packed_half_78),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_63 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_fragment_62 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_61 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_113 = !condition_1 ? matrix_fragment_88 : matrix_fragment_89;

    matrix_accumulator_115 = !condition_1 ? matrix_fragment_89 : matrix_fragment_88;

    packed_half_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_83), 0x3c203c20u);

    packed_half_79 = add_half_pairs(select_half_operand<2, false, false>(packed_half_79),
                                    select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_111 = condition_0 ? matrix_accumulator_110 : matrix_accumulator_111;

    packed_half_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_83), 0x3e473e47u);

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_103 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(packed_half_79),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_accumulator_140 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_140),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    matrix_accumulator_142 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_142),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_102 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    packed_half_79 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                                        0x3e473e47u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_accumulator_141 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_82 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_80 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    packed_half_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_143),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_81 = (packed_half_83 << 0x5u) + 0x7ff88000u;

    packed_half_83 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                    select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_99 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_98 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_112 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_accumulator_140 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_140), 0x3c203c20u);

    matrix_accumulator_142 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_142), 0x3c203c20u);

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    packed_half_79 = (packed_half_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(packed_half_83),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_accumulator_141 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141), 0x3c203c20u);

    packed_half_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x3c203c20u);

    matrix_accumulator_140 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_140), 0x3e473e47u);

    matrix_accumulator_142 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_142), 0x3e473e47u);

    packed_half_83 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                                        select_half_operand<2, false, false>(packed_half_79));

    matrix_accumulator_141 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141), 0x3e473e47u);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_30 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x3e473e47u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_90 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = address_word_2 * 0x4u;

    matrix_fragment_89 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(packed_half_83),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_92 = (matrix_accumulator_140 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = (matrix_accumulator_142 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_88 = (matrix_accumulator_141 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_91 = (packed_half_30 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_15 = matrix_fragment_14 & 0x1cu;

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                    select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                    select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_64 =
        uint32_t(int64_t(int32_t(address_word_2)) * (1ll << 0x1du) >> 32) + matrix_fragment_15;

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_14 = !condition_1 ? matrix_accumulator_111 : matrix_accumulator_114;

    matrix_fragment_15 = condition_0 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_fragment_67 = condition_0 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_accumulator_111 = !condition_1 ? matrix_accumulator_114 : matrix_accumulator_111;

    matrix_fragment_13 = 0x1u ^ matrix_fragment_64;

    matrix_fragment_12 = 0x2u ^ matrix_fragment_64;

    matrix_fragment_66 = 0x3u ^ matrix_fragment_64;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_113), "r"(matrix_fragment_64), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_113 = result_word;
    }

    matrix_fragment_96 = condition_0 ? matrix_fragment_65 : packed_half_30;

    matrix_accumulator_114 = condition_0 ? packed_half_30 : matrix_fragment_65;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_12), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_111), "r"(matrix_fragment_66), "r"(0x1fu),
              "r"(__activemask()));
        packed_half_30 = result_word;
    }

    matrix_fragment_97 = !condition_1 ? matrix_fragment_15 : matrix_fragment_96;

    matrix_accumulator_116 = !condition_1 ? matrix_fragment_67 : matrix_accumulator_114;

    matrix_accumulator_117 = !condition_1 ? matrix_fragment_96 : matrix_fragment_15;

    matrix_fragment_67 = !condition_1 ? matrix_accumulator_114 : matrix_fragment_67;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_97), "r"(matrix_fragment_64), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_13), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_116 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_117), "r"(matrix_fragment_12), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(matrix_fragment_66), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    matrix_fragment_15 = !condition_3 ? matrix_fragment_113 : matrix_fragment_14;

    matrix_fragment_14 = !condition_3 ? matrix_fragment_14 : matrix_fragment_113;

    matrix_fragment_96 = !condition_3 ? matrix_accumulator_115 : packed_half_30;

    matrix_accumulator_115 = !condition_3 ? packed_half_30 : matrix_accumulator_115;

    matrix_fragment_65 = !condition_4 ? matrix_fragment_15 : matrix_fragment_96;

    matrix_fragment_64 = !condition_4 ? matrix_fragment_14 : matrix_accumulator_115;

    matrix_fragment_15 = !condition_4 ? matrix_fragment_96 : matrix_fragment_15;

    packed_half_30 = uint32_t(int32_t(index_word_165) >> 0x2u);

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_13 = !condition_3 ? matrix_fragment_97 : matrix_accumulator_116;

    matrix_fragment_97 = !condition_3 ? matrix_accumulator_116 : matrix_fragment_97;

    matrix_fragment_67 = packed_half_30 + 0x8u;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = uint32_t(int32_t(index_word_165) >> 0x1fu);

    matrix_fragment_12 = !condition_3 ? matrix_accumulator_117 : matrix_accumulator_114;

    matrix_accumulator_114 = !condition_3 ? matrix_accumulator_114 : matrix_accumulator_117;

    matrix_fragment_96 = uint32_t(int32_t(matrix_fragment_67) >> 0x1fu);

    matrix_fragment_15 =
        uint32_t(combine_words(matrix_fragment_15, 0u) << 0x6u >> 32) + packed_half_30;

    matrix_fragment_65 = !condition_4 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_fragment_66 = !condition_4 ? matrix_fragment_97 : matrix_accumulator_114;

    matrix_fragment_12 = !condition_4 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_fragment_96 =
        uint32_t(combine_words(matrix_fragment_96, 0u) << 0x6u >> 32) + matrix_fragment_67;

    matrix_fragment_15 = matrix_fragment_15 & 0xffc0u;

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_14 = !condition_4 ? matrix_accumulator_115 : matrix_fragment_14;

    matrix_fragment_97 = !condition_4 ? matrix_accumulator_114 : matrix_fragment_97;

    matrix_fragment_96 = matrix_fragment_96 & 0xffc0u;

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_66 = packed_half_30 + (0u - matrix_fragment_15);

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_96 = matrix_fragment_67 + (0u - matrix_fragment_96);

    matrix_fragment_12 = permute_signed(matrix_fragment_66, 0u, 0x8880u);

    matrix_fragment_64 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_64),
                                        select_half_operand<1, false, false>(matrix_fragment_64));

    matrix_fragment_65 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_65),
                                        select_half_operand<1, false, false>(matrix_fragment_65));

    matrix_fragment_14 = permute_signed(matrix_fragment_96, 0u, 0x8880u);

    matrix_fragment_12 = permute_signed(matrix_fragment_12, 0u, 0x7710u);

    matrix_fragment_64 = permute_signed(matrix_fragment_64, matrix_fragment_65, 0x5410u);

    matrix_fragment_14 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_fragment_13 = uint32_t(uint32_t(matrix_fragment_12) >> 0xau);

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_14) >> 0xau);

    matrix_fragment_13 = matrix_fragment_13 & 0x1fu;

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x4100410u);

    matrix_fragment_15 = matrix_fragment_14 & 0x1fu;

    matrix_fragment_65 = matrix_fragment_66 + matrix_fragment_13;

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_64 & 65535u) * 65537u);

    matrix_fragment_97 = uint32_t(int64_t(int32_t(index_word_165)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_64 = half_to_float_bits((matrix_fragment_64 >> 16) * 65537u);

    matrix_fragment_14 = matrix_fragment_96 + matrix_fragment_15;

    matrix_fragment_13 = matrix_fragment_65 & 0xe0u;

    matrix_accumulator_114 = reciprocal_bits(matrix_fragment_12);

    matrix_accumulator_111 =
        uint32_t(int64_t(int32_t(index_word_165)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_accumulator_116 = uint32_t(int32_t(matrix_fragment_97) >> 0x1fu);

    matrix_fragment_67 = matrix_fragment_14 & 0xe0u;

    matrix_fragment_15 = reciprocal_bits(matrix_fragment_64);

    matrix_fragment_118 = uint32_t(int32_t(matrix_accumulator_111) >> 0x1fu);

    matrix_fragment_13 = 0u - matrix_fragment_13;

    matrix_accumulator_116 =
        uint32_t(combine_words(matrix_accumulator_116, 0u) << 0x6u >> 32) + matrix_fragment_97;

    matrix_fragment_118 =
        uint32_t(combine_words(matrix_fragment_118, 0u) << 0x6u >> 32) + matrix_accumulator_111;

    matrix_accumulator_115 = 0u - matrix_fragment_67;

    matrix_fragment_12 = matrix_accumulator_116 & 0xffc0u;

    matrix_fragment_67 = permute_signed(matrix_fragment_13, 0u, 0x7710u);

    matrix_fragment_118 = matrix_fragment_118 & 0xffc0u;

    matrix_fragment_12 = matrix_fragment_97 + (0u - matrix_fragment_12);

    matrix_fragment_66 = matrix_fragment_66 + matrix_fragment_67;

    matrix_fragment_13 = matrix_accumulator_111 + (0u - matrix_fragment_118);

    matrix_accumulator_114 = pack_half(matrix_fragment_15, matrix_accumulator_114);

    matrix_fragment_15 = permute_signed(matrix_fragment_12, 0u, 0x8880u);

    matrix_fragment_66 = matrix_fragment_66 & 0xffffu;

    matrix_accumulator_115 = permute_signed(matrix_accumulator_115, 0u, 0x7710u);

    matrix_fragment_64 = permute_signed(matrix_fragment_13, 0u, 0x8880u);

    matrix_fragment_15 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    matrix_fragment_67 = permute_signed(matrix_fragment_66, 0u, 0x8880u);

    matrix_fragment_96 = matrix_fragment_96 + matrix_accumulator_115;

    matrix_fragment_66 = permute_signed(matrix_fragment_64, 0u, 0x7710u);

    matrix_fragment_64 = uint32_t(uint32_t(matrix_fragment_15) >> 0xau);

    matrix_fragment_15 = matrix_fragment_67;

    matrix_fragment_96 = matrix_fragment_96 & 0xffffu;

    matrix_fragment_97 = permute_signed(matrix_fragment_96, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_113 = uint32_t(int64_t(int32_t(index_word_165)) * (1ll << 0x1eu) >> 32) + 0x20u;

    matrix_fragment_67 = matrix_fragment_64 & 0x1fu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_97), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    matrix_accumulator_116 = uint32_t(int32_t(matrix_fragment_113) >> 0x1fu);

    matrix_fragment_65 = matrix_fragment_65 & 0xffffu;

    matrix_fragment_67 = matrix_fragment_12 + matrix_fragment_67;

    matrix_fragment_66 = uint32_t(uint32_t(matrix_fragment_66) >> 0xau);

    matrix_accumulator_116 =
        uint32_t(combine_words(matrix_accumulator_116, 0u) << 0x6u >> 32) + matrix_fragment_113;

    matrix_fragment_64 = uint32_t(uint32_t(matrix_fragment_65) >> 0x5u);

    matrix_fragment_66 = matrix_fragment_66 & 0x1fu;

    matrix_accumulator_111 =
        uint32_t(int64_t(int32_t(index_word_165)) * (1ll << 0x1eu) >> 32) + 0x28u;

    matrix_fragment_96 = matrix_fragment_67 & 0xe0u;

    matrix_accumulator_116 = matrix_accumulator_116 & 0xffc0u;

    condition_0 = (matrix_fragment_64 & 0x1u) != 0;

    matrix_fragment_64 = uint32_t(int32_t(matrix_accumulator_111) >> 0x1fu);

    matrix_fragment_66 = matrix_fragment_13 + matrix_fragment_66;

    matrix_accumulator_115 = matrix_fragment_14 & 0xffffu;

    matrix_fragment_96 = 0u - matrix_fragment_96;

    matrix_fragment_65 = matrix_fragment_113 + (0u - matrix_accumulator_116);

    matrix_fragment_113 =
        uint32_t(combine_words(matrix_fragment_64, 0u) << 0x6u >> 32) + matrix_accumulator_111;

    matrix_fragment_64 = uint32_t(uint32_t(matrix_accumulator_115) >> 0x5u);

    matrix_fragment_14 = matrix_fragment_66 & 0xe0u;

    matrix_accumulator_115 = permute_signed(matrix_fragment_96, 0u, 0x7710u);

    matrix_fragment_96 = permute_signed(matrix_fragment_15, matrix_fragment_96, 0x7632u);

    condition_1 = (matrix_fragment_64 & 0x1u) != 0;

    matrix_fragment_118 = 0u - matrix_fragment_14;

    matrix_fragment_64 = matrix_fragment_113 & 0xffc0u;

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_44);

    matrix_fragment_113 = condition_0 ? matrix_fragment_96 : matrix_fragment_15;

    matrix_accumulator_116 = permute_signed(matrix_fragment_97, matrix_accumulator_116, 0x7632u);

    matrix_fragment_118 = permute_signed(matrix_fragment_118, 0u, 0x7710u);

    matrix_fragment_64 = matrix_accumulator_111 + (0u - matrix_fragment_64);

    packed_half_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_accumulator_111 = condition_1 ? matrix_accumulator_116 : matrix_fragment_97;

    matrix_accumulator_115 = matrix_fragment_12 + matrix_accumulator_115;

    matrix_accumulator_117 = matrix_fragment_13 + matrix_fragment_118;

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_accumulator_115 = matrix_accumulator_115 & 0xffffu;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = permute_signed(matrix_fragment_65, 0u, 0x8880u);

    matrix_fragment_97 = permute_signed(matrix_accumulator_115, 0u, 0x8880u);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_33 = permute_signed(matrix_fragment_33, 0u, 0x7710u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_96 = transpose_half_tile(matrix_fragment_41);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_33 = uint32_t(uint32_t(matrix_fragment_33) >> 0xau);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_97), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_40);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_46 = matrix_fragment_33 & 0x1fu;

    matrix_fragment_33 = transpose_half_tile(matrix_fragment_55);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_59 = transpose_half_tile(matrix_fragment_39);

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = transpose_half_tile(matrix_fragment_54);

    matrix_accumulator_117 = matrix_accumulator_117 & 0xffffu;

    matrix_fragment_58 = transpose_half_tile(matrix_fragment_38);

    matrix_fragment_67 = matrix_fragment_67 & 0xffffu;

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_accumulator_101 = permute_signed(matrix_accumulator_117, 0u, 0x8880u);

    matrix_fragment_67 = uint32_t(uint32_t(matrix_fragment_67) >> 0x5u);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_46 = matrix_fragment_65 + matrix_fragment_46;

    matrix_fragment_57 = permute_signed(matrix_fragment_64, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_accumulator_101), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_101 = result_word;
    }

    condition_0 = (matrix_fragment_67 & 0x1u) != 0;

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = permute_signed(matrix_fragment_97, matrix_fragment_56, 0x7632u);

    matrix_fragment_57 = permute_signed(matrix_fragment_57, 0u, 0x7710u);

    matrix_accumulator_100 = matrix_fragment_46 & 0xe0u;

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_35 = condition_0 ? matrix_fragment_56 : matrix_fragment_97;

    matrix_fragment_56 = uint32_t(uint32_t(matrix_fragment_57) >> 0xau);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 = 0u - matrix_accumulator_100;

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = matrix_fragment_66 & 0xffffu;

    matrix_fragment_67 = matrix_fragment_56 & 0x1fu;

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_38 = permute_signed(matrix_accumulator_100, 0u, 0x7710u);

    matrix_fragment_39 = uint32_t(uint32_t(matrix_fragment_66) >> 0x5u);

    matrix_fragment_96 = matrix_fragment_64 + matrix_fragment_67;

    matrix_fragment_65 = matrix_fragment_65 + matrix_fragment_38;

    condition_1 = (matrix_fragment_39 & 0x1u) != 0;

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_66 = permute_signed(matrix_accumulator_101, matrix_fragment_66, 0x7632u);

    matrix_fragment_65 = matrix_fragment_65 & 0xffffu;

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_33 = condition_1 ? matrix_fragment_66 : matrix_accumulator_101;

    matrix_fragment_65 = permute_signed(matrix_fragment_65, 0u, 0x8880u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_14 = matrix_fragment_96 & 0xe0u;

    matrix_fragment_48 = 0u - matrix_fragment_14;

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_73 = permute_signed(matrix_fragment_48, 0u, 0x7710u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_67 = uint32_t(int64_t(int32_t(index_word_165)) * (1ll << 0x1eu) >> 32) + 0x30u;

    matrix_fragment_48 = matrix_fragment_64 + matrix_fragment_73;

    matrix_fragment_73 = matrix_fragment_65;

    matrix_fragment_72 = uint32_t(int32_t(matrix_fragment_67) >> 0x1fu);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = matrix_fragment_48 & 0xffffu;

    index_word_165 = uint32_t(int64_t(int32_t(index_word_165)) * (1ll << 0x1eu) >> 32) + 0x38u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_73), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_fragment_72 =
        uint32_t(combine_words(matrix_fragment_72, 0u) << 0x6u >> 32) + matrix_fragment_67;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_74 = uint32_t(int32_t(index_word_165) >> 0x1fu);

    matrix_fragment_72 = matrix_fragment_72 & 0xffc0u;

    matrix_fragment_74 =
        uint32_t(combine_words(matrix_fragment_74, 0u) << 0x6u >> 32) + index_word_165;

    matrix_accumulator_100 = matrix_fragment_67 + (0u - matrix_fragment_72);

    matrix_fragment_74 = matrix_fragment_74 & 0xffc0u;

    matrix_fragment_46 = matrix_fragment_46 & 0xffffu;

    matrix_fragment_72 = permute_signed(matrix_accumulator_100, 0u, 0x8880u);

    index_word_165 = index_word_165 + (0u - matrix_fragment_74);

    matrix_fragment_46 = uint32_t(uint32_t(matrix_fragment_46) >> 0x5u);

    matrix_fragment_72 = permute_signed(matrix_fragment_72, 0u, 0x7710u);

    matrix_fragment_73 = permute_signed(index_word_165, 0u, 0x8880u);

    condition_0 = (matrix_fragment_46 & 0x1u) != 0;

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_46 = uint32_t(uint32_t(matrix_fragment_72) >> 0xau);

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_75 = permute_signed(matrix_fragment_48, matrix_fragment_75, 0x7632u);

    matrix_fragment_73 = permute_signed(matrix_fragment_73, 0u, 0x7710u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                          select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_97 = permute_signed(matrix_fragment_66, 0u, 0x8880u);

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_96 = matrix_fragment_96 & 0xffffu;

    matrix_accumulator_101 = matrix_fragment_46 & 0x1fu;

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = condition_0 ? matrix_fragment_75 : matrix_fragment_48;

    matrix_fragment_74 = uint32_t(uint32_t(matrix_fragment_73) >> 0xau);

    matrix_fragment_48 = uint32_t(uint32_t(matrix_fragment_96) >> 0x5u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_97), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    matrix_fragment_96 = matrix_accumulator_100 + matrix_accumulator_101;

    matrix_fragment_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                      select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    matrix_fragment_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_74 = matrix_fragment_74 & 0x1fu;

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    matrix_accumulator_101 = matrix_fragment_96 & 0xe0u;

    matrix_accumulator_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                           select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_86 = index_word_165 + matrix_fragment_74;

    matrix_accumulator_101 = 0u - matrix_accumulator_101;

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                          select_half_operand<0, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_78),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_69 = permute_signed(matrix_accumulator_101, 0u, 0x7710u);

    split_address_words(packed_half_106, matrix_accumulator_107,
                        uint64_t(uint32_t(address_word_160)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_accumulator_100 = matrix_accumulator_100 + matrix_fragment_69;

    matrix_fragment_69 = index_word_164 + 0x8u;

    condition_0 = (matrix_fragment_48 & 0x1u) != 0;

    index_word_164 = index_word_164 + 0xcu;

    matrix_fragment_48 = permute_signed(matrix_fragment_97, matrix_fragment_48, 0x7632u);

    matrix_accumulator_100 = matrix_accumulator_100 & 0xffffu;

    matrix_accumulator_101 = matrix_fragment_69 + address_word_2 + (0u - index_word_161);

    matrix_fragment_12 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_69 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_106, matrix_accumulator_107) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_accumulator_105 = index_word_164 + address_word_2 + (0u - index_word_161);

    matrix_fragment_48 = condition_0 ? matrix_fragment_48 : matrix_fragment_97;

    matrix_accumulator_109 = permute_signed(matrix_accumulator_100, 0u, 0x8880u);

    split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                        uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    packed_half_78 = matrix_fragment_96 & 0xffffu;

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    split_address_words(packed_half_104, matrix_accumulator_105,
                        uint64_t(uint32_t(matrix_accumulator_105)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_71 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_fragment_85 = matrix_fragment_86 & 0xe0u;

    matrix_accumulator_107 = matrix_fragment_86 & 0xffffu;

    matrix_fragment_86 = uint32_t(uint32_t(packed_half_78) >> 0x5u);

    packed_half_78 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_104, matrix_accumulator_105) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    matrix_fragment_85 = 0u - matrix_fragment_85;

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    matrix_accumulator_76 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                      select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_accumulator_109), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_109 = result_word;
    }

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_76 = permute_signed(matrix_fragment_85, 0u, 0x7710u);

    index_word_165 = index_word_165 + matrix_accumulator_76;

    index_word_165 = index_word_165 & 0xffffu;

    matrix_accumulator_115 = permute_signed(index_word_165, 0u, 0x8880u);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_accumulator_115), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    condition_0 = (matrix_fragment_86 & 0x1u) != 0;

    packed_half_106 = permute_signed(matrix_accumulator_109, packed_half_106, 0x7632u);

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(0u));

    packed_half_106 = condition_0 ? packed_half_106 : matrix_accumulator_109;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(packed_half_106));

    matrix_accumulator_107 = uint32_t(uint32_t(matrix_accumulator_107) >> 0x5u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    condition_1 = (matrix_accumulator_107 & 0x1u) != 0;

    matrix_fragment_86 = permute_signed(matrix_accumulator_115, matrix_fragment_86, 0x7632u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_86 = condition_1 ? matrix_fragment_86 : matrix_accumulator_115;

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<0, false, false>(packed_half_106));

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                            select_half_operand<0, false, false>(packed_half_106));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_7);

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(packed_half_106));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_5 = transpose_half_tile(matrix_fragment_5);

    matrix_fragment_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                      select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_21 = transpose_half_tile(matrix_fragment_21);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = transpose_half_tile(matrix_fragment_4);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_20 = transpose_half_tile(matrix_fragment_20);

    matrix_fragment_9 = transpose_half_tile(matrix_fragment_9);

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_23 = transpose_half_tile(matrix_fragment_23);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_25 = transpose_half_tile(matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_11 = transpose_half_tile(matrix_fragment_11);

    matrix_fragment_27 = transpose_half_tile(matrix_fragment_27);

    matrix_fragment_8 = transpose_half_tile(matrix_fragment_8);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    packed_half_3 = transpose_half_tile(matrix_fragment_6);

    matrix_fragment_22 = transpose_half_tile(matrix_fragment_22);

    matrix_fragment_24 = transpose_half_tile(matrix_fragment_24);

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_26);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_10);

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<0, false, false>(matrix_fragment_113));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_113),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                      select_half_operand<2, false, false>(matrix_fragment_113));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(matrix_fragment_35));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_79),
                            select_half_operand<0, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<0, false, false>(matrix_fragment_33));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<0, false, false>(matrix_fragment_48));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(packed_half_106));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(packed_half_106));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    packed_half_83 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_83),
                                         select_half_operand<0, false, false>(packed_half_106));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<0, false, false>(packed_half_106));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_83),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_54,
                             matrix_fragment_55);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_44,
                             matrix_fragment_45);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x2000u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_40,
                             matrix_fragment_41);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x3000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x1000u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_139 = threadIdx.y;

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                       select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x1000u, matrix_fragment_64,
                                         matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x2000u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_0 + 0x3000u, matrix_fragment_44,
                                         matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_76 = unpack_fp8_bits(matrix_fragment_16);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_21 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_accumulator_31 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_68 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_70 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_42 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_43 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_20 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        uint64_t(uint32_t(matrix_accumulator_139)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(packed_half_152));

    tile_word_0 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        combine_words(matrix_fragment_36, matrix_fragment_37) + 0x98440u);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_7 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(tile_word_0),
                                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(packed_half_78));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(packed_half_78));

    uniform_index_word_4 = 0u;

    __syncthreads();
    do {
        split_address_words(matrix_fragment_38, matrix_fragment_39,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_36, matrix_fragment_37));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_38, matrix_fragment_39) - 0x200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_38, matrix_fragment_39),
                                            matrix_fragment_44, matrix_fragment_45,
                                            matrix_fragment_46, matrix_fragment_47, memory_ranges,
                                            error_counts);

        tile_word_0 = (address_word_2 << 0x4u) + uniform_index_word_5;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_6 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_fragment_36, matrix_fragment_37,
                            combine_words(matrix_fragment_36, matrix_fragment_37) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_0, matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
            matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_5 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_0 + 0x1000u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_6;

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_0 + 0x2000u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, tile_word_0 + 0x3000u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_68,
                                 matrix_fragment_69);

        multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_70,
                                 matrix_fragment_71);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_26,
                                 matrix_fragment_27);
    } while (!uniform_condition_0);
    uniform_index_word_5 = blockIdx.x;

    uniform_parameter_word_6 = parameters[parameter_optional_control_10];
    uniform_parameter_word_7 = parameters[parameter_scalar_control_11];

    tile_word_0 = uint32_t(combine_words(packed_half_30, 0u) << 0x2u >> 32) + packed_half_30;

    matrix_fragment_64 = threadIdx.y;

    matrix_accumulator_31 = address_word_2;

    packed_half_3 = tile_word_0 & 0xfffffffcu;

    uniform_index_word_4 = blockIdx.y;

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = parameters[parameter_optional_control_20];
    matrix_fragment_37 = parameters[parameter_optional_control_21];

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_5 = (uniform_index_word_5 << 0x3u) + uniform_parameter_word_6;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = packed_half_30 + uniform_index_word_5 + (0u - packed_half_3);

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_parameter_word_7;

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = matrix_fragment_64 * 0x2u;

    matrix_fragment_44 = matrix_fragment_38 + 0x4u;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = uint32_t(int64_t(int32_t(matrix_accumulator_31)) * (1ll << 0x1cu) >> 32) +
                         uniform_index_word_4;

    condition_0 = int32_t(matrix_fragment_36) > int32_t(0u);

    condition_1 = int32_t(matrix_fragment_37) > int32_t(0u);

    matrix_fragment_41 = matrix_fragment_39 + 0x1u;

    matrix_fragment_53 = matrix_fragment_43 + 0x6u;

    matrix_fragment_45 = matrix_fragment_43 + 0x2u;

    tile_word_0 = condition_0 ? matrix_fragment_36 : address_word_28;

    packed_half_3 = condition_1 ? matrix_fragment_37 : index_word_29;

    index_word_29 = index_word_29 & 0xfffffffeu;

    matrix_fragment_51 = matrix_fragment_43 + 0x4u;

    condition_0 = int32_t(matrix_fragment_43) >= int32_t(tile_word_0);

    condition_4 = int32_t(matrix_fragment_38) >= int32_t(packed_half_3);

    condition_1 = int32_t(matrix_fragment_53) >= int32_t(tile_word_0);

    condition_6 = int32_t(matrix_fragment_44) >= int32_t(packed_half_3);

    condition_2 = int32_t(matrix_fragment_45) >= int32_t(tile_word_0);

    condition_0 = int32_t(matrix_fragment_43) < int32_t(0u) || condition_0;

    condition_4 = int32_t(matrix_fragment_38) < int32_t(0u) || condition_4;

    condition_1 = int32_t(matrix_fragment_53) < int32_t(0u) || condition_1;

    condition_6 = int32_t(matrix_fragment_44) < int32_t(0u) || condition_6;

    condition_5 = int32_t(matrix_fragment_51) >= int32_t(tile_word_0);

    condition_2 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_2;

    condition_3 = condition_0 | condition_4;

    packed_half_3 = packed_half_3 * 0x4u;

    condition_0 = condition_0 | condition_6;

    condition_5 = int32_t(matrix_fragment_51) < int32_t(0u) || condition_5;

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = uint32_t(condition_0);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    if (condition_3) {
        index_word_29 = 0x1u | index_word_29;
    }

    condition_3 = condition_5 | condition_4;

    index_word_29 = index_word_29 & 0xfffffffdu;

    condition_5 = condition_5 | condition_6;

    if (condition_1) {
        index_word_29 = 0x2u | index_word_29;
    }

    condition_1 = condition_2 | condition_4;

    condition_0 = (index_word_29 & 0x1u) != 0;

    condition_2 = condition_2 | condition_6;

    if (!condition_3) {
        matrix_fragment_32 = tile_word_0 * matrix_fragment_39 + matrix_fragment_51;
    }

    address_word_28 = uint32_t(condition_0);

    if (!condition_3) {
        matrix_fragment_10 = matrix_fragment_38 * 0x4u;
    }

    condition_0 = (index_word_29 & 0x2u) != 0;

    if (!condition_5) {
        matrix_fragment_22 = parameters[parameter_output_low_2];
        matrix_fragment_23 = parameters[parameter_output_high_3];
    }

    if (!condition_3) {
        matrix_fragment_61 = permute_signed(matrix_fragment_61, matrix_fragment_24, 0x5410u);

        matrix_fragment_32 = packed_half_3 * matrix_fragment_32 + matrix_fragment_10;
    }

    condition_4 = condition_0 | condition_4;

    if (!condition_1) {
        matrix_fragment_47 = matrix_fragment_38 * 0x4u;
    }

    condition_6 = condition_0 | condition_6;

    if (!condition_1) {
        matrix_fragment_40 = tile_word_0 * matrix_fragment_39 + matrix_fragment_45;
    }

    condition_0 = int32_t(address_word_28) != int32_t(0u);

    if (!condition_1) {
        packed_half_30 = parameters[parameter_output_low_2];
        matrix_accumulator_31 = parameters[parameter_output_high_3];
    }

    if (!condition_5) {
        matrix_fragment_65 = permute_signed(matrix_fragment_65, matrix_fragment_68, 0x5410u);
    }

    if (!condition_1) {
        matrix_fragment_40 = packed_half_3 * matrix_fragment_40 + matrix_fragment_47;

        matrix_fragment_47 = matrix_fragment_40 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_2) {
        matrix_fragment_40 = tile_word_0 * matrix_fragment_39 + matrix_fragment_45;
    }

    if (!condition_4) {
        matrix_fragment_11 = matrix_fragment_38 * 0x4u;
    }

    if (!condition_0) {
        address_word_28 = parameters[parameter_output_low_2];
        index_word_29 = parameters[parameter_output_high_3];

        matrix_fragment_6 = matrix_fragment_38 * 0x4u;

        matrix_fragment_55 = permute_signed(matrix_fragment_55, matrix_fragment_46, 0x5410u);

        matrix_fragment_14 = tile_word_0 * matrix_fragment_39 + matrix_fragment_43;
    }

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        matrix_fragment_48 = matrix_fragment_38 * 0x4u;

        matrix_fragment_42 = tile_word_0 * matrix_fragment_41 + matrix_fragment_43;

        matrix_fragment_36 = parameters[parameter_output_low_2];
        matrix_fragment_37 = parameters[parameter_output_high_3];

        matrix_fragment_6 = packed_half_3 * matrix_fragment_14 + matrix_fragment_6;

        matrix_fragment_42 = packed_half_3 * matrix_fragment_42 + matrix_fragment_48;
    }

    if (!condition_1) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_47)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_30, matrix_accumulator_31));
    }

    if (!condition_0) {
        matrix_fragment_7 = matrix_fragment_6 + address_word_2 + (0u - index_word_161);

        matrix_fragment_49 = matrix_fragment_42 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_5) {
        matrix_fragment_42 = tile_word_0 * matrix_fragment_39 + matrix_fragment_51;
    }

    if (!condition_1) {
        matrix_accumulator_31 = permute_signed(matrix_fragment_46, matrix_fragment_57, 0x5410u);
    }

    if (!condition_6) {
        matrix_fragment_10 = matrix_fragment_44 * 0x4u;
    }

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_8 = matrix_fragment_44 * 0x4u;
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, index_word_29));
    }

    if (!condition_3) {
        matrix_fragment_57 = permute_signed(matrix_fragment_57, matrix_fragment_16, 0x5410u);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_fragment_55, memory_ranges, error_counts);

        split_address_words(address_word_28, index_word_29,
                            int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_36, matrix_fragment_37));

        matrix_fragment_37 = permute_signed(matrix_fragment_18, matrix_fragment_59, 0x5410u);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_14, matrix_fragment_15),
                                    matrix_accumulator_31, memory_ranges, error_counts);
    }

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_18 = tile_word_0 * matrix_fragment_41 + matrix_fragment_45;
    }

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(address_word_28, index_word_29),
                                    matrix_fragment_37, memory_ranges, error_counts);
    }

    condition_0 = int32_t(matrix_fragment_50) != int32_t(0u);

    if (!condition_2) {
        matrix_fragment_45 = tile_word_0 * matrix_fragment_41 + matrix_fragment_45;
    }

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_fragment_36 = tile_word_0 * matrix_fragment_39 + matrix_fragment_53;
    }

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_fragment_14 = tile_word_0 * matrix_fragment_41 + matrix_fragment_51;
    }

    if (!condition_5) {
        matrix_fragment_15 = tile_word_0 * matrix_fragment_41 + matrix_fragment_51;
    }

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_5 = matrix_fragment_38 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_55 = permute_signed(matrix_fragment_55, matrix_fragment_6, 0x5410u);
    }

    if (!condition_0) {
        packed_half_30 = tile_word_0 * matrix_fragment_39 + matrix_fragment_43;
    }

    if (!condition_2) {
        matrix_fragment_51 = permute_signed(matrix_fragment_51, matrix_fragment_4, 0x5410u);
    }

    if (!condition_0) {
        matrix_fragment_43 = tile_word_0 * matrix_fragment_41 + matrix_fragment_43;
    }

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        address_word_28 = tile_word_0 * matrix_fragment_41 + matrix_fragment_53;
    }

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_18 = packed_half_3 * matrix_fragment_18 + matrix_fragment_5;
    }

    if (!condition_4) {
        matrix_fragment_63 = permute_signed(matrix_fragment_63, matrix_fragment_4, 0x5410u);
    }

    if (!condition_6) {
        matrix_fragment_39 = tile_word_0 * matrix_fragment_39 + matrix_fragment_53;
    }

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_fragment_41 = tile_word_0 * matrix_fragment_41 + matrix_fragment_53;

        matrix_fragment_71 = permute_signed(matrix_fragment_71, matrix_fragment_6, 0x5410u);
    }

    if (!condition_2) {
        matrix_accumulator_31 = packed_half_3 * matrix_fragment_45 + matrix_fragment_8;
    }

    if (!condition_6) {
        matrix_fragment_67 = permute_signed(matrix_fragment_67, matrix_fragment_4, 0x5410u);
    }

    if (!condition_0) {
        matrix_fragment_6 = parameters[parameter_output_low_2];
        matrix_fragment_7 = parameters[parameter_output_high_3];
    }

    tile_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_fragment_36 = packed_half_3 * matrix_fragment_36 + matrix_fragment_11;
    }

    if (!condition_0) {
        matrix_fragment_49 = permute_signed(matrix_fragment_49, matrix_fragment_12, 0x5410u);
    }

    if (!condition_2) {
        matrix_fragment_12 = matrix_fragment_44 * 0x4u;
    }

    if (!condition_1) {
        matrix_fragment_47 = permute_signed(matrix_fragment_47, tile_word_0, 0x5410u);
    }

    tile_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_4 = parameters[parameter_output_low_2];
        matrix_fragment_5 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_12 = packed_half_3 * matrix_fragment_40 + matrix_fragment_12;
    }

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_fragment_40 = matrix_fragment_44 * 0x4u;
    }

    if (!condition_4) {
        matrix_fragment_59 = permute_signed(matrix_fragment_59, tile_word_0, 0x5410u);
    }

    if (!condition_0) {
        tile_word_0 = matrix_fragment_44 * 0x4u;

        matrix_fragment_53 = permute_signed(matrix_fragment_53, matrix_fragment_34, 0x5410u);
    }

    if (!condition_5) {
        matrix_fragment_40 = packed_half_3 * matrix_fragment_42 + matrix_fragment_40;
    }

    if (!condition_2) {
        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    if (!condition_0) {
        tile_word_0 = packed_half_3 * packed_half_30 + tile_word_0;
    }

    if (!condition_1) {
        matrix_fragment_11 = matrix_fragment_18 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_0) {
        packed_half_30 = matrix_fragment_44 * 0x4u;
    }

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_fragment_34 = matrix_fragment_38 * 0x4u;
    }

    if (!condition_0) {
        matrix_fragment_13 = tile_word_0 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_5) {
        matrix_fragment_42 = matrix_fragment_44 * 0x4u;
    }

    if (!condition_4) {
        matrix_fragment_16 = parameters[parameter_output_low_2];
        matrix_fragment_17 = parameters[parameter_output_high_3];

        matrix_fragment_38 = matrix_fragment_38 * 0x4u;
    }

    if (!condition_5) {
        matrix_fragment_69 = permute_signed(matrix_fragment_69, matrix_fragment_26, 0x5410u);
    }

    if (!condition_6) {
        matrix_fragment_44 = matrix_fragment_44 * 0x4u;
    }

    if (!condition_2) {
        matrix_accumulator_31 = matrix_accumulator_31 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_0) {
        packed_half_30 = packed_half_3 * matrix_fragment_43 + packed_half_30;
    }

    if (!condition_3) {
        matrix_fragment_33 = matrix_fragment_32 + address_word_2 + (0u - index_word_161);

        matrix_fragment_34 = packed_half_3 * matrix_fragment_14 + matrix_fragment_34;

        matrix_fragment_18 = parameters[parameter_output_low_2];
        matrix_fragment_19 = parameters[parameter_output_high_3];
    }

    if (!condition_4) {
        matrix_fragment_38 = packed_half_3 * address_word_28 + matrix_fragment_38;

        matrix_fragment_35 = matrix_fragment_36 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_6) {
        tile_word_0 = packed_half_3 * matrix_fragment_39 + matrix_fragment_10;
    }

    if (!condition_3) {
        matrix_fragment_37 = matrix_fragment_34 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_5) {
        matrix_fragment_42 = packed_half_3 * matrix_fragment_15 + matrix_fragment_42;
    }

    if (!condition_4) {
        matrix_fragment_39 = matrix_fragment_38 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_6) {
        matrix_fragment_44 = packed_half_3 * matrix_fragment_41 + matrix_fragment_44;
    }

    if (!condition_2) {
        packed_half_3 = matrix_fragment_12 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_1) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (!condition_3) {
        matrix_fragment_14 = parameters[parameter_output_low_2];
        matrix_fragment_15 = parameters[parameter_output_high_3];
    }

    if (!condition_5) {
        matrix_fragment_41 = matrix_fragment_40 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_6) {
        matrix_fragment_43 = tile_word_0 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_fragment_47, memory_ranges, error_counts);
    }

    if (!condition_5) {
        matrix_fragment_45 = matrix_fragment_42 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_0) {
        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];

        packed_half_3 = packed_half_30 + address_word_2 + (0u - index_word_161);

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_fragment_49, memory_ranges, error_counts);
    }

    if (!condition_6) {
        index_word_161 = matrix_fragment_44 + address_word_2 + (0u - index_word_161);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_51, memory_ranges, error_counts);

        matrix_fragment_12 = parameters[parameter_output_low_2];
        matrix_fragment_13 = parameters[parameter_output_high_3];
    }

    if (!condition_5) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_41)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));
    }

    if (!condition_4) {
        matrix_fragment_20 = parameters[parameter_output_low_2];
        matrix_fragment_21 = parameters[parameter_output_high_3];
    }

    if (!condition_6) {
        matrix_fragment_24 = parameters[parameter_output_low_2];
        matrix_fragment_25 = parameters[parameter_output_high_3];
    }

    if (!condition_0) {
        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

        pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3),
                                    matrix_fragment_53, memory_ranges, error_counts);
    }

    if (!condition_5) {
        matrix_fragment_26 = parameters[parameter_output_low_2];
        matrix_fragment_27 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_accumulator_31)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));
    }

    if (!condition_3) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_33)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_fragment_55, memory_ranges, error_counts);
    }

    if (!condition_6) {
        address_word_28 = parameters[parameter_output_low_2];
        index_word_29 = parameters[parameter_output_high_3];
    }

    if (!condition_4) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_35)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_12, matrix_fragment_13),
                                    matrix_fragment_57, memory_ranges, error_counts);

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_fragment_37)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_18, matrix_fragment_19));
    }

    if (!condition_4) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_14, matrix_fragment_15),
                                    matrix_fragment_59, memory_ranges, error_counts);

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                    matrix_fragment_61, memory_ranges, error_counts);
    }

    if (!condition_6) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));
    }

    if (!condition_4) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_fragment_63, memory_ranges, error_counts);
    }

    if (!condition_5) {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(matrix_fragment_45)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_26, matrix_fragment_27));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_fragment_65, memory_ranges, error_counts);
    }

    if (!condition_6) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_67, memory_ranges, error_counts);

        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(index_word_161)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_28, index_word_29));
    }

    if (!condition_5) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_18, matrix_fragment_19),
                                    matrix_fragment_69, memory_ranges, error_counts);
    }

    if (!condition_6) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_20, matrix_fragment_21),
                                    matrix_fragment_71, memory_ranges, error_counts);
    }

    __syncthreads();
    return;
}
extern "C" __global__ __maxnreg__(255) void phoenix_window_attention_channels_256_output_view(
    KernelArguments parameters) {
    run_window_attention_channels_256_output_view<false>(parameters.words, nullptr, nullptr);
}
