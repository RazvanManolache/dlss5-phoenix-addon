// DLSS5 Phoenix optimized kernel: window_attention/channels_64/output_view.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (__CUDA_ARCH__ >= 900)
#define DLSS5_TILE_SHARED_BYTES 5120
#include "../../../common/device_memory_aligned.cuh"
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
__device__ __forceinline__ void run_window_attention_channels_64_output_view(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t packed_half_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t packed_half_2 = 0;
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
    uint32_t matrix_accumulator_24 = 0;
    uint32_t matrix_accumulator_25 = 0;
    uint32_t matrix_accumulator_26 = 0;
    uint32_t matrix_accumulator_27 = 0;
    uint32_t matrix_fragment_28 = 0;
    uint32_t matrix_fragment_29 = 0;
    uint32_t matrix_fragment_30 = 0;
    uint32_t matrix_fragment_31 = 0;
    uint32_t matrix_accumulator_32 = 0;
    uint32_t matrix_accumulator_33 = 0;
    uint32_t matrix_accumulator_34 = 0;
    uint32_t matrix_accumulator_35 = 0;
    uint32_t matrix_fragment_36 = 0;
    uint32_t matrix_fragment_37 = 0;
    uint32_t matrix_accumulator_38 = 0;
    uint32_t matrix_accumulator_39 = 0;
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
    uint32_t matrix_fragment_78 = 0;
    uint32_t matrix_fragment_79 = 0;
    uint32_t matrix_fragment_80 = 0;
    uint32_t matrix_fragment_81 = 0;
    uint32_t matrix_fragment_82 = 0;
    uint32_t matrix_fragment_83 = 0;
    uint32_t matrix_fragment_84 = 0;
    uint32_t matrix_fragment_85 = 0;
    uint32_t packed_half_86 = 0;
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
    uint32_t matrix_accumulator_98 = 0;
    uint32_t matrix_accumulator_99 = 0;
    uint32_t matrix_accumulator_100 = 0;
    uint32_t matrix_accumulator_101 = 0;
    uint32_t matrix_accumulator_102 = 0;
    uint32_t matrix_accumulator_103 = 0;
    uint32_t packed_half_104 = 0;
    uint32_t packed_half_105 = 0;
    uint32_t matrix_accumulator_106 = 0;
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
    uint32_t matrix_accumulator_118 = 0;
    uint32_t matrix_accumulator_119 = 0;
    uint32_t matrix_fragment_120 = 0;
    uint32_t matrix_fragment_121 = 0;
    uint32_t matrix_fragment_122 = 0;
    uint32_t matrix_fragment_123 = 0;
    uint32_t packed_half_124 = 0;
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
    uint32_t packed_half_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t matrix_accumulator_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t matrix_accumulator_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
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
    uint32_t matrix_fragment_154 = 0;
    uint32_t matrix_fragment_155 = 0;
    uint32_t index_word_156 = 0;
    uint32_t index_word_157 = 0;
    uint32_t matrix_accumulator_158 = 0;
    uint32_t matrix_accumulator_159 = 0;
    uint32_t address_word_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t index_word_163 = 0;
    uint32_t address_word_164 = 0;
    uint32_t address_word_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_block_index_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_parameter_word_8 = 0;
    uint32_t uniform_reused_word_9 = 0;
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

    uniform_parameter_word_8 = parameters[parameter_optional_control_15];

    matrix_fragment_8 = parameters[parameter_optional_control_10];
    matrix_fragment_9 = parameters[parameter_scalar_control_11];

    packed_half_3 = blockIdx.x;

    uniform_index_word_6 = parameters[parameter_tensor_height_8];

    matrix_fragment_5 = threadIdx.x;

    uniform_parameter_word_7 = parameters[parameter_scalar_control_14];

    matrix_fragment_10 = threadIdx.y;

    uniform_block_index_5 = parameters[parameter_tensor_width_9];

    matrix_fragment_7 = threadIdx.z;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_8));

    uniform_index_word_6 = 0u - uniform_index_word_4 + uniform_index_word_6;

    uniform_condition_1 = uint32_t(uniform_index_word_6) != 0u;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_7));

    packed_half_0 = packed_half_0 * 0x8u + matrix_fragment_9;

    uniform_block_index_5 = 0u - uniform_index_word_4 + uniform_block_index_5;

    matrix_fragment_4 = packed_half_3 * 0x8u + matrix_fragment_8;

    packed_half_2 = packed_half_0 + (0u - uniform_parameter_word_8);

    if (uniform_condition_1) {
        uniform_index_word_6 = uniform_index_word_6 + 0xffffffffu;
    }

    uniform_index_word_4 = 0u;

    uniform_condition_0 = uint32_t(uniform_block_index_5) != 0u;

    matrix_fragment_6 = matrix_fragment_4 + (0u - uniform_parameter_word_7);

    packed_half_3 = packed_half_2 + 0x7u;

    if (uniform_condition_1) {
        uniform_parameter_word_7 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);
    }

    uniform_parameter_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    packed_half_3 = uint32_t(combine_words(packed_half_3, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_1) {
        uniform_parameter_word_7 =
            uint32_t(combine_words(uniform_parameter_word_7, 0u) << 0x3u >> 32) +
            uniform_index_word_6;
    }

    condition_0 = (matrix_fragment_10 | matrix_fragment_5 | matrix_fragment_7) != 0;

    matrix_fragment_5 = packed_half_2 + 0x8u;

    if (uniform_condition_1) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    packed_half_2 = max(int32_t(0u), int32_t(packed_half_2));

    matrix_fragment_7 = matrix_fragment_6 + 0x8u;

    if (uniform_condition_0) {
        uniform_block_index_5 = uniform_block_index_5 + 0xffffffffu;
    }

    condition_1 = int32_t(matrix_fragment_5) > int32_t(0u);

    packed_half_2 = uint32_t(uint32_t(packed_half_2) >> 0x3u);

    if (uniform_condition_0) {
        uniform_index_word_6 = uint32_t(int32_t(uniform_block_index_5) >> 0x1fu);
    }

    condition_2 = int32_t(matrix_fragment_7) > int32_t(0u);

    matrix_fragment_5 = condition_1 ? packed_half_3 : 0u;

    matrix_fragment_7 = matrix_fragment_6 + 0x7u;

    packed_half_3 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_index_word_6 =
            uint32_t(combine_words(uniform_index_word_6, 0u) << 0x3u >> 32) + uniform_block_index_5;
    }

    matrix_fragment_6 = max(int32_t(0u), int32_t(matrix_fragment_6));

    matrix_fragment_11 = min(int32_t(matrix_fragment_5), int32_t(uniform_index_word_4));

    matrix_fragment_7 = uint32_t(combine_words(matrix_fragment_7, 0u) << 0x1du >> 32) + 0x1u;

    uniform_index_word_4 = 0u;

    packed_half_3 = uint32_t(combine_words(packed_half_3, 0u) << 0x2u >> 32) + matrix_fragment_4;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    condition_0 = int32_t(packed_half_2) >= int32_t(matrix_fragment_11) || condition_0;

    matrix_fragment_7 = condition_2 ? matrix_fragment_7 : 0u;

    matrix_fragment_5 = uint32_t(int32_t(packed_half_3) >> 0x2u);
    if (!condition_0) {
        matrix_fragment_13 = min(int32_t(matrix_fragment_7), int32_t(uniform_index_word_4));

        matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_6) >> 0x3u);

        matrix_fragment_7 = packed_half_2;
        do {
            condition_0 = int32_t(matrix_fragment_6) >= int32_t(matrix_fragment_13);
            if (!condition_0) {
                matrix_fragment_15 = matrix_fragment_7 * uniform_index_word_4;

                matrix_fragment_8 = matrix_fragment_6;
                do {
                    packed_half_2 = parameters[parameter_wait_optional_low_12];
                    packed_half_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_9 = matrix_fragment_15 + matrix_fragment_8;

                    split_address_words(packed_half_2, packed_half_3,
                                        int64_t(int32_t(matrix_fragment_9)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(packed_half_2, packed_half_3));

                    matrix_fragment_9 = pipeline_read<CheckBounds>(
                        combine_words(packed_half_2, packed_half_3), memory_ranges, error_counts);

                    matrix_fragment_8 = matrix_fragment_8 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_8) != uint32_t(matrix_fragment_13);

                    condition_0 = int32_t(matrix_fragment_9) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_9 = pipeline_read<CheckBounds>(
                                combine_words(packed_half_2, packed_half_3), memory_ranges,
                                error_counts);

                            condition_0 = int32_t(matrix_fragment_9) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            matrix_fragment_7 = matrix_fragment_7 + 0x1u;

            condition_0 = uint32_t(matrix_fragment_7) != uint32_t(matrix_fragment_11);
        } while (condition_0);
    }
    index_word_156 = parameters[parameter_tensor_height_8];
    index_word_157 = parameters[parameter_tensor_width_9];

    matrix_fragment_7 = uint32_t(int32_t(packed_half_0) >> 0x1fu);

    address_word_161 = threadIdx.x & 31u;

    condition_6 = false;

    __syncwarp(__activemask());

    matrix_fragment_7 =
        uint32_t(combine_words(matrix_fragment_7, 0u) << 0x2u >> 32) + packed_half_0;

    address_word_164 = parameters[parameter_trained_weights_low_4];
    address_word_165 = parameters[parameter_trained_weights_high_5];

    condition_5 = false;

    matrix_fragment_7 =
        uint32_t(int64_t(int32_t(matrix_fragment_7)) * (1ll << 0x1eu) >> 32) + matrix_fragment_10;

    matrix_fragment_11 = matrix_fragment_5 + 0x1u;

    matrix_fragment_13 = 0u;

    packed_half_2 = index_word_156 & 0xfffffffcu;

    packed_half_3 = uint32_t(int32_t(index_word_156) >> 0x1fu);

    packed_half_0 = uint32_t(int32_t(index_word_157) >> 0x1fu);

    __syncthreads();

    condition_0 = uint32_t(packed_half_2) != uint32_t(0x4u);

    packed_half_3 = uint32_t(combine_words(packed_half_3, 0u) << 0x2u >> 32) + index_word_156;

    packed_half_0 = uint32_t(combine_words(packed_half_0, 0u) << 0x2u >> 32) + index_word_157;

    packed_half_3 = uint32_t(int32_t(packed_half_3) >> 0x2u);

    packed_half_0 = uint32_t(int32_t(packed_half_0) >> 0x2u);

    matrix_fragment_10 = index_word_157 & 0xfffffffcu;

    condition_3 = int32_t(matrix_fragment_5) >= int32_t(packed_half_0);

    if (condition_0) {
        condition_1 = int32_t(matrix_fragment_7) >= int32_t(packed_half_3);

        condition_4 = int32_t(matrix_fragment_7) >= int32_t(packed_half_3);

        matrix_fragment_13 = packed_half_0 * matrix_fragment_7;
    }

    condition_3 = int32_t(matrix_fragment_4) > int32_t(0u - 0x4u) && !condition_3;

    if (condition_0) {
        condition_6 = int32_t(matrix_fragment_7) < int32_t(0u) || condition_1;
    }

    condition_1 = true;

    if (condition_0) {
        condition_5 = int32_t(matrix_fragment_7) < int32_t(0u) || condition_4;
    }

    condition_2 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_6;

    if (condition_0) {
        condition_1 = !condition_6;
    }

    matrix_fragment_6 = uint32_t(condition_5) << 5;

    condition_1 = condition_1 & (condition_2 | condition_3);

    condition_2 = true;

    matrix_fragment_9 = uint32_t(condition_1) << 1;

    condition_1 = int32_t(matrix_fragment_6) != int32_t(0u);

    if (condition_0) {
        condition_5 = int32_t(matrix_fragment_7) >= int32_t(packed_half_3);
    }

    packed_half_2 = condition_6 ? 0x40u : 0u;

    condition_4 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_1;

    if (condition_0) {
        condition_2 = !condition_1;
    }

    matrix_fragment_8 = uint32_t(condition_1) << 1;

    condition_2 = condition_2 & (condition_4 | condition_3);

    if (condition_0) {
        condition_4 = int32_t(matrix_fragment_7) >= int32_t(packed_half_3);
    }

    condition_3 = false;

    condition_1 = int32_t(packed_half_2) != int32_t(0u);

    packed_half_2 = uint32_t(condition_2) << 2;

    if (condition_0) {
        condition_3 = int32_t(matrix_fragment_7) < int32_t(0u) || condition_4;
    }

    condition_4 = false;

    if (condition_0) {
        condition_4 = int32_t(matrix_fragment_7) < int32_t(0u) || condition_5;
    }

    condition_5 = int32_t(matrix_fragment_11) >= int32_t(packed_half_0);

    condition_6 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_3;

    matrix_fragment_6 = uint32_t(condition_1) << 1;

    condition_1 = uint32_t(matrix_fragment_10) == uint32_t(0x4u) || condition_4;

    condition_2 = int32_t(matrix_fragment_4) > int32_t(0u - 0x8u) && !condition_5;

    condition_5 = true;

    if (condition_0) {
        condition_5 = !condition_3;
    }

    matrix_fragment_16 = condition_3 ? matrix_fragment_11 : 0u;

    condition_3 = uint32_t(matrix_fragment_10) != uint32_t(0x4u);

    condition_5 = condition_5 & (condition_6 | condition_2);

    condition_6 = true;

    if (condition_0) {
        condition_6 = !condition_4;
    }

    matrix_fragment_16 = !condition_3 ? matrix_fragment_16 : matrix_fragment_11;

    matrix_fragment_4 = 0u;

    condition_6 = condition_6 & (condition_1 | condition_2);

    if (condition_0) {
        matrix_fragment_4 = packed_half_0 * matrix_fragment_7;
    }

    condition_1 = int32_t(matrix_fragment_6) != int32_t(0u);

    condition_2 = int32_t(packed_half_2) != int32_t(0u);

    matrix_fragment_10 = 0u;

    if (condition_5) {
        matrix_fragment_14 = parameters[parameter_input_low_0];
        matrix_fragment_15 = parameters[parameter_input_high_1];
    }

    if (condition_0) {
        matrix_fragment_10 = packed_half_0 * matrix_fragment_7;
    }

    matrix_fragment_6 = condition_1 ? matrix_fragment_5 : 0u;

    condition_1 = int32_t(matrix_fragment_8) != int32_t(0u);

    if (condition_5) {
        matrix_fragment_13 = matrix_fragment_16 + matrix_fragment_13;
    }

    if (!condition_3) {
        matrix_fragment_11 = condition_4 ? matrix_fragment_11 : 0u;
    }

    matrix_fragment_17 = !condition_3 ? matrix_fragment_6 : matrix_fragment_5;

    matrix_fragment_12 = condition_1 ? matrix_fragment_5 : 0u;

    condition_1 = int32_t(matrix_fragment_9) != int32_t(0u);

    if (condition_6) {
        matrix_fragment_22 = parameters[parameter_input_low_0];
        matrix_fragment_23 = parameters[parameter_input_high_1];
    }

    matrix_fragment_6 = 0u;

    if (condition_5) {
        matrix_fragment_13 = matrix_fragment_13 * 0x100u;
    }

    matrix_fragment_5 = !condition_3 ? matrix_fragment_12 : matrix_fragment_5;

    if (condition_0) {
        matrix_fragment_6 = packed_half_0 * matrix_fragment_7;
    }

    if (condition_2) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];

        matrix_fragment_10 = matrix_fragment_5 + matrix_fragment_10;
    }

    if (condition_6) {
        packed_half_0 = matrix_fragment_11 + matrix_fragment_6;
    }

    if (!condition_5) {
        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_1) {
        matrix_fragment_4 = matrix_fragment_17 + matrix_fragment_4;
    }

    if (condition_2) {
        matrix_fragment_7 = matrix_fragment_10 * 0x100u;
    }

    if (!condition_1) {
        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_1) {
        packed_half_2 = parameters[parameter_input_low_0];
        packed_half_3 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_1) {
        matrix_fragment_19 = permute_signed(matrix_fragment_19, matrix_fragment_19, 0x5410u);
    }

    if (condition_1) {
        matrix_fragment_5 = matrix_fragment_4 * 0x100u;
    }

    if (!condition_2) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                          select_half_operand<2, false, false>(0u));
    }

    if (condition_5) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    matrix_accumulator_24 = address_word_161;

    if (!condition_6) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_6) {
        matrix_accumulator_25 = packed_half_0 * 0x100u;
    }

    if (!condition_1) {
        matrix_fragment_16 = matrix_fragment_19;

        matrix_fragment_17 = matrix_fragment_19;

        matrix_fragment_18 = matrix_fragment_19;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_2) {
        matrix_fragment_7 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5410u);
    }

    if (condition_1) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    if (!condition_2) {
        matrix_fragment_4 = matrix_fragment_7;

        matrix_fragment_5 = matrix_fragment_7;

        matrix_fragment_6 = matrix_fragment_7;
    }

    if (!condition_5) {
        matrix_fragment_11 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    index_word_163 = matrix_accumulator_24 & 0xfffffffcu;

    if (condition_1) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    if (!condition_6) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    if (condition_6) {
        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));
    }

    if (!condition_5) {
        matrix_fragment_8 = matrix_fragment_11;
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(packed_half_2, packed_half_3), matrix_fragment_16, matrix_fragment_17,
            matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);
    }

    if (!condition_5) {
        matrix_fragment_9 = matrix_fragment_11;

        matrix_fragment_10 = matrix_fragment_11;
    }

    if (!condition_6) {
        matrix_fragment_14 = matrix_fragment_15;
    }

    if (condition_6) {
        split_address_words(matrix_fragment_22, matrix_fragment_23,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_22, matrix_fragment_23));
    }

    if (!condition_6) {
        matrix_fragment_12 = matrix_fragment_15;

        matrix_fragment_13 = matrix_fragment_15;
    }

    matrix_accumulator_27 = address_word_161 + 0xcu + (0u - index_word_163);

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    packed_half_3 = 0u - index_word_163 + address_word_161;

    matrix_accumulator_25 = address_word_161 + 0x8u + (0u - index_word_163);

    if (condition_6) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_22, matrix_fragment_23) + 0x200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    split_address_words(packed_half_2, packed_half_3,
                        int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_fragment_21 = address_word_161 + 0x4u + (0u - index_word_163);

    matrix_accumulator_109 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, packed_half_3) + 0x7010ull, memory_ranges, error_counts);

    split_address_words(matrix_fragment_22, matrix_fragment_23,
                        uint64_t(uint32_t(matrix_accumulator_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_27 = address_word_161 + 0x10u + (0u - index_word_163);

    matrix_fragment_31 = address_word_161 + 0x18u + (0u - index_word_163);

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        uint64_t(uint32_t(matrix_fragment_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_33 = address_word_161 + 0x1cu + (0u - index_word_163);

    matrix_fragment_22 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x7010ull, memory_ranges,
        error_counts);

    packed_half_3 = address_word_161 + 0x14u + (0u - index_word_163);

    split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                        uint64_t(uint32_t(matrix_accumulator_25)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_fragment_28 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_20, matrix_fragment_21) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                        uint64_t(uint32_t(matrix_accumulator_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_fragment_29 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_24, matrix_accumulator_25) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(packed_half_2, packed_half_3,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_27 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_26, matrix_accumulator_27) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        uint64_t(uint32_t(matrix_fragment_31)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_32 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, packed_half_3) + 0x7010ull, memory_ranges, error_counts);

    split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                        uint64_t(uint32_t(matrix_accumulator_33)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_33 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_20, matrix_fragment_21) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_24, matrix_accumulator_25) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_accumulator_102 = 0u;

    uniform_condition_0 = true;

    packed_half_146 = unpack_fp8_bits(matrix_fragment_4);

    packed_half_144 = unpack_fp8_bits(matrix_fragment_5);

    packed_half_149 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    packed_half_147 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_154 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    packed_half_151 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    packed_half_150 = unpack_fp8_bits(matrix_fragment_18);

    matrix_accumulator_111 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    packed_half_0 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    packed_half_148 = unpack_fp8_bits(matrix_fragment_19);

    matrix_accumulator_140 = unpack_fp8_bits(matrix_fragment_7);

    matrix_accumulator_139 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_accumulator_138 = unpack_fp8_bits(matrix_fragment_8);

    matrix_accumulator_137 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_113 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_112 = unpack_fp8_bits(matrix_fragment_10);

    matrix_accumulator_110 = unpack_fp8_bits(matrix_fragment_11);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_12);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_13);

    packed_half_145 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_accumulator_143 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    packed_half_136 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_7 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_155 = unpack_fp8_bits(matrix_fragment_16);

    packed_half_152 = unpack_fp8_bits(matrix_fragment_17);

    matrix_accumulator_142 = unpack_fp8_bits(matrix_fragment_6);

    matrix_accumulator_141 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_154),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_155),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_40));
    do {
        packed_half_105 = matrix_accumulator_102 * 0x800u;

        split_address_words(packed_half_104, packed_half_105,
                            uint64_t(uint32_t(packed_half_105)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_164, address_word_165));

        split_address_words(packed_half_104, packed_half_105,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_104, packed_half_105));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105), matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x200ull, matrix_fragment_96,
            matrix_fragment_97, matrix_accumulator_98, matrix_accumulator_99, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1000ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x400ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x600ull, matrix_fragment_88,
            matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1200ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1600ull, matrix_fragment_92,
            matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x800ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1400ull, matrix_fragment_84,
            matrix_fragment_85, packed_half_86, matrix_fragment_87, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1800ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        matrix_accumulator_101 = matrix_accumulator_102 * 0x400u;

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_164, address_word_165));

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4000ull,
            matrix_fragment_64, matrix_fragment_65, matrix_accumulator_66, matrix_accumulator_67,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4200ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_154),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_113),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_155),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_112),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_120 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<2, false, false>(0u));

        packed_half_153 = 0x2b28u;

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_48, matrix_fragment_49, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_48, matrix_fragment_49, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                           select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_50, matrix_fragment_51, 0u, 0u);

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        multiply_accumulate_half(packed_half_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_96, matrix_fragment_97, 0u, 0u);

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                          select_half_operand<2, false, false>(matrix_fragment_120));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_96, matrix_fragment_97, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_122, matrix_fragment_123, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_78, matrix_fragment_79, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_78, matrix_fragment_79, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_76, matrix_fragment_77, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_76, matrix_fragment_77, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_72, matrix_fragment_73, packed_half_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_120,
                                 matrix_fragment_121);

        multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_122,
                                 matrix_fragment_123);

        matrix_fragment_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_fragment_75 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_92, matrix_fragment_93, matrix_fragment_78,
                                 matrix_fragment_79);

        matrix_fragment_78 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_114,
                                 matrix_accumulator_115);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_78),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_98,
                                 matrix_accumulator_99);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_56, matrix_fragment_57, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_56, matrix_fragment_57, 0u, 0u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_106), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_107), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_130,
                                 matrix_accumulator_131);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_132,
                                 matrix_accumulator_133);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75), 0xc400c400u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_74),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_fragment_75), 0x3b283b28u);

        matrix_fragment_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0x44004400u);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78),
            select_half_operand<2, false, false>(matrix_fragment_79), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_76), 0x44004400u);

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_77), 0x44004400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_78 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0xc400c400u);

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_75 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73), 0x44004400u);

        matrix_accumulator_116 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_fragment_79));

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        matrix_fragment_74 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_78),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75), 0xc400c400u);

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_74),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_79 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78),
            select_half_operand<2, false, false>(matrix_fragment_79), 0x3b283b28u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_75 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74),
            select_half_operand<2, false, false>(matrix_fragment_75), 0x3b283b28u);

        matrix_fragment_79 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                select_half_operand<2, false, false>(matrix_fragment_79));

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_75 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_77 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_94,
                                 matrix_fragment_95);

        matrix_fragment_78 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_fragment_78));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_92,
                                 matrix_fragment_93);

        matrix_fragment_61 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_81), 0x44004400u);

        matrix_fragment_76 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_86, matrix_fragment_87, matrix_accumulator_134,
                                 matrix_accumulator_135);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_83), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 packed_half_86, matrix_fragment_87, matrix_accumulator_118,
                                 matrix_accumulator_119);

        packed_half_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        packed_half_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(packed_half_86), 0x3b283b28u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        packed_half_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                select_half_operand<2, false, false>(packed_half_86));

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_84), 0x3b283b28u);

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        packed_half_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_86),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                           select_half_operand<2, false, false>(packed_half_86));

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4400ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_81 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_120), 0x44004400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_80),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_81 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_81), 0xc400c400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80),
            select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

        matrix_fragment_82 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_84 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_81),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_80 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_121), 0x44004400u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_81),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80), 0xc400c400u);

        matrix_fragment_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_80),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80),
            select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        matrix_fragment_56 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_fragment_81));

        matrix_fragment_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                                select_half_operand<2, false, false>(matrix_fragment_80));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0xa00ull, matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4600ull,
            matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
            memory_ranges, error_counts);

        packed_half_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_121 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_122 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                          select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_58, matrix_fragment_59, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_58, matrix_fragment_59, 0u, 0u);

        matrix_fragment_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_120),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_58 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                          select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_86),
                          select_half_operand<2, false, false>(matrix_fragment_121));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1a00ull, matrix_fragment_84,
            matrix_fragment_85, packed_half_86, matrix_fragment_87, memory_ranges, error_counts);

        matrix_fragment_57 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_fragment_122));

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_122, matrix_fragment_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(packed_half_124, matrix_accumulator_125, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_132), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130), 0x44004400u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_60,
                                 matrix_fragment_61);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x44004400u);

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_59 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_accumulator_103), 0x3b283b28u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_69 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_133), 0x44004400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_96,
                                 matrix_fragment_97);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_accumulator_130 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4a00ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_57 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_57),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4800ull,
            matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
            memory_ranges, error_counts);

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_130),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130),
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x3b283b28u);

        matrix_fragment_70 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_fragment_90 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_accumulator_131));

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0xc400c400u);

        matrix_fragment_91 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x44004400u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0xc400c400u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_90 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_accumulator_103 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98), 0x44004400u);

        matrix_accumulator_130 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0xc400c400u);

        matrix_accumulator_103 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_115), 0x44004400u);

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_130),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0xc400c400u);

        matrix_accumulator_131 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130),
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x3b283b28u);

        matrix_accumulator_130 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_130 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_accumulator_130), 0x3b283b28u);

        matrix_accumulator_103 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<2, false, false>(matrix_accumulator_131));

        matrix_accumulator_98 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_127 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_90));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                          select_half_operand<2, false, false>(matrix_accumulator_98));

        matrix_accumulator_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_99), 0x44004400u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_accumulator_114 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98), 0xc400c400u);

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                          select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_accumulator_130));

        matrix_accumulator_114 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_98),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_accumulator_115 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_114 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98),
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x3b283b28u);

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                          select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_115 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_accumulator_126 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                          select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_accumulator_115 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_103 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_114 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(matrix_accumulator_115));

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(matrix_accumulator_126));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                          select_half_operand<2, false, false>(matrix_accumulator_114));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_64,
                                 matrix_fragment_65);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_66,
                                 matrix_accumulator_67);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0xc00ull, matrix_fragment_64,
            matrix_fragment_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1c00ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0xe00ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_122,
                                 matrix_fragment_123);

        multiply_accumulate_half(packed_half_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, packed_half_124,
                                 matrix_accumulator_125);

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_104, packed_half_105) + 0x1e00ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_80, matrix_fragment_81, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_80, matrix_fragment_81, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_82, matrix_fragment_83, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_122, matrix_fragment_123, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_82, matrix_fragment_83, 0u, 0u);

        matrix_fragment_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

        matrix_fragment_82 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_88,
                                 matrix_fragment_89);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_82),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82),
            select_half_operand<2, false, false>(matrix_fragment_83), 0x3b283b28u);

        multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_90,
                                 matrix_fragment_91);

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_120), 0x44004400u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_86, matrix_fragment_87, matrix_fragment_80,
                                 matrix_fragment_81);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        matrix_fragment_82 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82), 0xc400c400u);

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_121), 0x44004400u);

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_82),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_84 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82),
            select_half_operand<2, false, false>(matrix_fragment_84), 0x3b283b28u);

        matrix_fragment_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                select_half_operand<2, false, false>(matrix_fragment_84));

        matrix_fragment_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(matrix_fragment_82), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_122, matrix_fragment_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 packed_half_86, matrix_fragment_87, matrix_fragment_122,
                                 matrix_fragment_123);

        matrix_fragment_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_80), 0x44004400u);

        matrix_fragment_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                                select_half_operand<2, false, false>(matrix_fragment_82));

        matrix_fragment_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        packed_half_86 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_81), 0x44004400u);

        matrix_fragment_85 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        matrix_fragment_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_fragment_83 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_fragment_82 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82), 0xc400c400u);

        matrix_fragment_87 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_86), 0xc400c400u);

        packed_half_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_85),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_83), 0xc400c400u);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_82),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        packed_half_86 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85),
            select_half_operand<2, false, false>(packed_half_86), 0x3b283b28u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_87),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_82),
            select_half_operand<2, false, false>(matrix_fragment_83), 0x3b283b28u);

        matrix_fragment_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_87),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        packed_half_86 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                select_half_operand<2, false, false>(packed_half_86));

        matrix_fragment_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_82), 0x3b283b28u);

        matrix_fragment_85 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_84 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0x44004400u);

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_82));

        matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_86),
                                           select_half_operand<2, false, false>(0u));

        packed_half_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_85 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

        matrix_fragment_84 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

        matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                           select_half_operand<2, false, false>(packed_half_86));

        packed_half_86 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85), 0xc400c400u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_84),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_85 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84),
            select_half_operand<2, false, false>(matrix_fragment_85), 0x3b283b28u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_fragment_83 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_120),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                select_half_operand<2, false, false>(matrix_fragment_85));

        matrix_fragment_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93), 0x44004400u);

        matrix_fragment_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                          select_half_operand<2, false, false>(matrix_fragment_83));

        matrix_fragment_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

        matrix_fragment_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0xc400c400u);

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                          select_half_operand<2, false, false>(matrix_fragment_80));

        matrix_fragment_95 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0xc400c400u);

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_92),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(matrix_fragment_87));

        matrix_fragment_87 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_86),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4c00ull,
            matrix_fragment_80, matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
            memory_ranges, error_counts);

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92),
            select_half_operand<2, false, false>(matrix_fragment_94), 0x3b283b28u);

        matrix_fragment_92 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x44004400u);

        matrix_accumulator_133 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_86),
            select_half_operand<2, false, false>(matrix_fragment_87), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4e00ull,
            matrix_fragment_84, matrix_fragment_85, packed_half_86, matrix_fragment_87,
            memory_ranges, error_counts);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                select_half_operand<2, false, false>(matrix_fragment_94));

        matrix_fragment_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0xc400c400u);

        matrix_fragment_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_122), 0x44004400u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_92),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0xc400c400u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92),
            select_half_operand<2, false, false>(matrix_fragment_93), 0x3b283b28u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_94),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94),
            select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

        matrix_fragment_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122),
                                select_half_operand<2, false, false>(matrix_fragment_92));

        matrix_fragment_92 =
            minimum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_106,
                                 matrix_accumulator_107);

        matrix_accumulator_101 = matrix_accumulator_102 * 0x200u;

        matrix_fragment_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0xc400c400u);

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_93 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_123), 0x44004400u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                select_half_operand<2, false, false>(matrix_accumulator_133));

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_164, address_word_165));

        packed_half_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_92),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_93 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93), 0xc400c400u);

        matrix_fragment_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_95),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        packed_half_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92),
            select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        matrix_fragment_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95),
            select_half_operand<2, false, false>(matrix_fragment_96), 0x3b283b28u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_93),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_98,
                                 matrix_accumulator_99);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                select_half_operand<2, false, false>(matrix_fragment_96));

        matrix_accumulator_158 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93),
            select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6400ull,
            matrix_fragment_96, matrix_fragment_97, matrix_accumulator_98, matrix_accumulator_99,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6200ull,
            matrix_fragment_92, matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6000ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6600ull,
            matrix_accumulator_100, matrix_accumulator_101, matrix_accumulator_102,
            matrix_accumulator_103, memory_ranges, error_counts);

        matrix_accumulator_158 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123),
                                select_half_operand<2, false, false>(matrix_accumulator_158));

        matrix_accumulator_133 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                select_half_operand<2, false, false>(packed_half_104));

        packed_half_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_123 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_122),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_158 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_158),
                          select_half_operand<2, false, false>(0u));

        packed_half_104 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(matrix_accumulator_133));

        packed_half_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(packed_half_105));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(matrix_fragment_123));

        matrix_accumulator_107 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(matrix_accumulator_158));

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, packed_half_104,
                                 packed_half_105, matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, packed_half_104,
                                 packed_half_105, matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, packed_half_104,
                                 packed_half_105, matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_64,
                                 matrix_fragment_65);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_60,
                                 matrix_fragment_61);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_40,
                                 matrix_fragment_41);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_66,
                                 matrix_accumulator_67);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_68,
                                 matrix_fragment_69);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_66), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

        matrix_fragment_59 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

        multiply_accumulate_half(packed_half_124, matrix_accumulator_125, packed_half_104,
                                 packed_half_105, matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_fragment_62, matrix_fragment_63, packed_half_124,
                                 matrix_accumulator_125);

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_65 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_67), 0x44004400u);

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

        matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

        packed_half_153 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_153), 0x37283728u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_accumulator_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_accumulator_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_48));

        packed_half_153 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(packed_half_153), 0x3b283b28u);

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                          select_half_operand<2, false, false>(0u));

        packed_half_153 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                select_half_operand<2, false, false>(packed_half_153));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 packed_half_86, matrix_fragment_87, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_120,
                                 matrix_fragment_121);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(packed_half_124, matrix_accumulator_125, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 packed_half_86, matrix_fragment_87, packed_half_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_120),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_121),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_124),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                          select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_38,
                                 matrix_accumulator_39);

        uniform_condition_1 = uniform_condition_0;

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_92, matrix_fragment_93, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_92, matrix_fragment_93, matrix_fragment_16,
                                 matrix_fragment_17);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_94, matrix_fragment_95, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_94, matrix_fragment_95, matrix_fragment_20,
                                 matrix_fragment_21);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_96, matrix_fragment_97, matrix_fragment_22,
                                 matrix_fragment_23);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_96, matrix_fragment_97, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_98, matrix_accumulator_99,
                                 matrix_accumulator_26, matrix_accumulator_27);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_34, matrix_accumulator_35);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_32, matrix_accumulator_33);

        uniform_condition_0 = false;

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_36,
                                 matrix_fragment_37);

        matrix_accumulator_102 = 0x1u;
    } while (uniform_condition_1);
    matrix_fragment_81 = threadIdx.y;

    uniform_block_index_5 = 0u;

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_4 = 0x400u;

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_6 = parameters[parameter_optional_control_10];
    uniform_parameter_word_7 = parameters[parameter_scalar_control_11];

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(packed_half_0));

    uniform_index_word_4 = (uniform_block_index_5 << 0x18u) + uniform_index_word_4;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                  select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = matrix_fragment_81 * 0x800u;

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_111 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_6 = matrix_fragment_9 + uniform_index_word_4;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                  select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = address_word_161 * 0x10u + matrix_fragment_6;

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8, matrix_fragment_44,
                                         matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_110 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                           select_half_operand<2, false, false>(packed_half_3));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_8 + 0x200u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_3 = matrix_fragment_81 * 0x300u;

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(packed_half_0));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_8 + 0x400u, matrix_accumulator_108, matrix_accumulator_109,
        matrix_accumulator_110, matrix_accumulator_111, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(matrix_fragment_7));

    split_address_words(packed_half_2, packed_half_3,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(matrix_fragment_4));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_8 + 0x600u, matrix_accumulator_24, matrix_accumulator_25,
        matrix_accumulator_26, matrix_accumulator_27, shared_bytes, memory_ranges, error_counts);

    split_address_words(packed_half_2, packed_half_3,
                        int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_2, packed_half_3));

    __syncthreads();

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x78a0ull,
                                        matrix_fragment_56, matrix_fragment_57, matrix_fragment_58,
                                        matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x72a0ull,
                                        matrix_fragment_28, matrix_fragment_29, matrix_fragment_30,
                                        matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x70a0ull,
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x74a0ull,
                                        matrix_fragment_4, matrix_fragment_5, matrix_fragment_6,
                                        matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x76a0ull,
                                        matrix_fragment_96, matrix_fragment_97,
                                        matrix_accumulator_98, matrix_accumulator_99, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x7aa0ull,
                                        matrix_fragment_64, matrix_fragment_65,
                                        matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x90a0ull,
                                        matrix_fragment_52, matrix_fragment_53, matrix_fragment_54,
                                        matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x8aa0ull,
                                        matrix_accumulator_32, matrix_accumulator_33,
                                        matrix_accumulator_34, matrix_accumulator_35, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x88a0ull,
                                        matrix_fragment_36, matrix_fragment_37,
                                        matrix_accumulator_38, matrix_accumulator_39, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x8ca0ull,
                                        packed_half_124, matrix_accumulator_125,
                                        matrix_accumulator_126, matrix_accumulator_127,
                                        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x8ea0ull,
                                        matrix_fragment_92, matrix_fragment_93, matrix_fragment_94,
                                        matrix_fragment_95, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3) + 0x92a0ull,
                                        matrix_fragment_60, matrix_fragment_61, matrix_fragment_62,
                                        matrix_fragment_63, memory_ranges, error_counts);

    packed_half_0 = (address_word_161 << 0x4u) + uniform_index_word_4;

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x400u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x200u, matrix_fragment_20,
                                        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0 + 0x600u, matrix_fragment_120, matrix_fragment_121,
        matrix_fragment_122, matrix_fragment_123, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x800u, matrix_fragment_48,
                                        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_16, matrix_fragment_17,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_56,
                             matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_158, matrix_accumulator_159, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_96, matrix_fragment_97, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_112, matrix_fragment_113, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_96, matrix_fragment_97, 0u, 0u);

    multiply_accumulate_half(packed_half_136, matrix_accumulator_137, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_52, matrix_fragment_53, packed_half_2, packed_half_3);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_20, matrix_fragment_21,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_32,
                             matrix_accumulator_33, matrix_accumulator_24, matrix_accumulator_25);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_10,
                             matrix_fragment_11);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                            select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xc00u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             packed_half_124, matrix_accumulator_125, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_158, matrix_accumulator_159, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_158,
                             matrix_accumulator_159);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_70,
                             matrix_fragment_71);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_34, matrix_accumulator_35, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_fragment_112, matrix_fragment_113, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             packed_half_124, matrix_accumulator_125, matrix_fragment_112,
                             matrix_fragment_113);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_130,
                             matrix_accumulator_131);

    multiply_accumulate_half(packed_half_136, matrix_accumulator_137, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_94, matrix_fragment_95, packed_half_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                            select_half_operand<2, false, false>(packed_half_3));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xa00u, matrix_fragment_68,
                                        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        uint64_t(uint32_t(matrix_fragment_81)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    address_word_160 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0xe0a0ull, memory_ranges,
        error_counts);

    matrix_fragment_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_76 = result_word;
    }

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_159),
                            select_half_operand<2, false, false>(matrix_accumulator_159));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_158),
                            select_half_operand<2, false, false>(matrix_accumulator_158));

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_96, matrix_fragment_97, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    matrix_fragment_84 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_74 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    matrix_fragment_73 = permute_signed(matrix_fragment_74, matrix_fragment_74, 0x5432u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_76));

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xe00u, matrix_fragment_72,
                                        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    split_address_words(matrix_fragment_122, matrix_fragment_123,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_fragment_9 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x4100410u);

    matrix_fragment_84 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_106 = result_word;
    }

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_77 & 65535u) * 65537u);

    matrix_fragment_85 = half_to_float_bits((matrix_fragment_77 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    packed_half_105 = half_to_float_bits((matrix_fragment_80 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_accumulator_140 = half_to_float_bits((matrix_fragment_80 >> 16) * 65537u);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_128,
                             matrix_accumulator_129);

    matrix_fragment_85 = rsqrt_bits(matrix_fragment_85);

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_114,
                             matrix_accumulator_115);

    packed_half_105 = rsqrt_bits(packed_half_105);

    matrix_fragment_82 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                             select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_accumulator_131));

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_124, matrix_accumulator_125, matrix_accumulator_134,
                             matrix_accumulator_135);

    matrix_accumulator_140 = rsqrt_bits(matrix_accumulator_140);

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_132,
                             matrix_accumulator_133);

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                      select_half_operand<2, false, false>(matrix_fragment_112),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                      select_half_operand<2, false, false>(matrix_fragment_113),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_83 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_fragment_83));

    packed_half_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                    select_half_operand<2, false, false>(matrix_fragment_80));

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                     select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_86), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_107 = result_word;
    }

    multiply_accumulate_half(packed_half_124, matrix_accumulator_125, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             packed_half_124, matrix_accumulator_125, matrix_fragment_4,
                             matrix_fragment_5);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_104), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_4 = pack_half(matrix_fragment_85, matrix_fragment_8);

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_8 = pack_half(matrix_accumulator_140, packed_half_105);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_87 = permute_signed(matrix_accumulator_106, matrix_accumulator_106, 0x5432u);

    packed_half_105 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_fragment_87));

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                     select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_7 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_143 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_106), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_96, matrix_fragment_97, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_105), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_142 = result_word;
    }

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                       select_half_operand<2, false, false>(matrix_fragment_9));

    packed_half_145 = half_to_float_bits((matrix_accumulator_143 >> 16) * 65537u);

    matrix_fragment_96 = half_to_float_bits((matrix_accumulator_143 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_143 = result_word;
    }

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_86),
                       select_half_operand<2, false, false>(matrix_accumulator_107));

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                     select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5432u);

    matrix_fragment_96 = rsqrt_bits(matrix_fragment_96);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_107), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_144 = result_word;
    }

    split_address_words(matrix_fragment_122, matrix_fragment_123,
                        int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_122, matrix_fragment_123));

    __syncthreads();

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_145 = rsqrt_bits(packed_half_145);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_104), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_97 = result_word;
    }

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x4100410u);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                     select_half_operand<2, false, false>(matrix_accumulator_142));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xa2a0ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_140,
                             matrix_accumulator_141);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_88 & 65535u) * 65537u);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_accumulator_143));

    matrix_fragment_11 = half_to_float_bits((matrix_fragment_88 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_105), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_92 = result_word;
    }

    matrix_fragment_96 = pack_half(packed_half_145, matrix_fragment_96);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xa4a0ull, matrix_fragment_84,
        matrix_fragment_85, packed_half_86, matrix_fragment_87, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    packed_half_145 = rsqrt_bits(matrix_fragment_11);

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(packed_half_144));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xa0a0ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_fragment_97 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    matrix_fragment_88 = permute_signed(matrix_accumulator_107, matrix_accumulator_107, 0x5432u);

    matrix_accumulator_98 = permute_signed(matrix_fragment_97, matrix_fragment_97, 0x5432u);

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                        select_half_operand<2, false, false>(matrix_fragment_92));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xa6a0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                       select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_fragment_8 = pack_half(packed_half_145, matrix_fragment_8);

    packed_half_146 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_107 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_107), 0x4100410u);

    matrix_accumulator_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x4100410u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_159 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_159),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = permute_signed(matrix_fragment_92, matrix_fragment_92, 0x5432u);

    packed_half_104 = half_to_float_bits((matrix_accumulator_107 & 65535u) * 65537u);

    matrix_fragment_97 = half_to_float_bits((matrix_accumulator_107 >> 16) * 65537u);

    packed_half_145 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    packed_half_147 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    packed_half_144 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_158),
                            select_half_operand<2, false, false>(matrix_fragment_96));

    packed_half_104 = rsqrt_bits(packed_half_104);

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    packed_half_151 = half_to_float_bits((matrix_accumulator_98 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_142,
                             matrix_accumulator_143);

    packed_half_152 = half_to_float_bits((matrix_accumulator_98 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xa8a0ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    matrix_fragment_97 = rsqrt_bits(matrix_fragment_97);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x4100410u);

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    packed_half_148 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_26));

    packed_half_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    packed_half_153 = half_to_float_bits((packed_half_105 & 65535u) * 65537u);

    packed_half_150 = permute_signed(matrix_fragment_13, matrix_fragment_13, 0x5432u);

    matrix_accumulator_106 = half_to_float_bits((packed_half_105 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xaca0ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    packed_half_153 = rsqrt_bits(packed_half_153);

    matrix_accumulator_106 = rsqrt_bits(matrix_accumulator_106);

    packed_half_104 = pack_half(matrix_fragment_97, packed_half_104);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xaaa0ull, matrix_fragment_96,
        matrix_fragment_97, matrix_accumulator_98, matrix_accumulator_99, memory_ranges,
        error_counts);

    packed_half_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_accumulator_107 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_107));

    packed_half_105 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(packed_half_105));

    packed_half_148 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(packed_half_148));

    packed_half_149 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_25),
                                      select_half_operand<2, false, false>(packed_half_149));

    packed_half_150 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                     select_half_operand<2, false, false>(packed_half_150));

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                     select_half_operand<2, false, false>(packed_half_148));

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(packed_half_105));

    packed_half_153 = pack_half(matrix_accumulator_106, packed_half_153);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_104));

    packed_half_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                          select_half_operand<2, false, false>(packed_half_104));

    load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_122, matrix_fragment_123) +
                                            0xaea0ull,
                                        packed_half_104, packed_half_105, matrix_accumulator_106,
                                        matrix_accumulator_107, memory_ranges, error_counts);

    packed_half_151 = rsqrt_bits(packed_half_151);

    packed_half_150 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_150), 0x4100410u);

    packed_half_152 = rsqrt_bits(packed_half_152);

    matrix_fragment_112 = half_to_float_bits((packed_half_150 & 65535u) * 65537u);

    matrix_fragment_154 = half_to_float_bits((packed_half_150 >> 16) * 65537u);

    matrix_fragment_112 = rsqrt_bits(matrix_fragment_112);

    matrix_fragment_154 = rsqrt_bits(matrix_fragment_154);

    packed_half_151 = pack_half(packed_half_152, packed_half_151);

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(packed_half_153));

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_153));

    packed_half_150 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                          select_half_operand<2, false, false>(packed_half_151));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_151));

    packed_half_151 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_153));

    packed_half_153 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_142));

    matrix_fragment_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    matrix_accumulator_114 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_113 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_124),
                                      select_half_operand<2, false, false>(packed_half_124),
                                      select_half_operand<2, false, false>(matrix_fragment_113));

    matrix_accumulator_158 = pack_half(matrix_fragment_154, matrix_fragment_112);

    matrix_fragment_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_fragment_113));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_149), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_112), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_113 = result_word;
    }

    matrix_fragment_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_158));

    matrix_accumulator_129 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_158));

    matrix_accumulator_128 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_158));

    matrix_accumulator_158 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_158));

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                      select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_accumulator_143));

    packed_half_144 = pack_half(0u, address_word_160);

    matrix_accumulator_128 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                      select_half_operand<2, false, false>(matrix_accumulator_128));

    packed_half_145 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                      select_half_operand<2, false, false>(matrix_accumulator_127),
                                      select_half_operand<2, false, false>(matrix_accumulator_115));

    packed_half_149 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_accumulator_141));

    matrix_accumulator_114 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_accumulator_114));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_148), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_9 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                        select_half_operand<2, false, false>(matrix_fragment_113));

    matrix_fragment_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_113),
                                        select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_113));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_fragment_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<0, false, false>(packed_half_144));

    packed_half_148 = add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                     select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_114 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(0u));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<0, false, false>(packed_half_144));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<0, false, false>(packed_half_144));

    matrix_accumulator_114 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                      select_half_operand<2, false, false>(matrix_accumulator_114));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_148), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                           select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                  select_half_operand<2, false, false>(matrix_fragment_12));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_159),
                                  select_half_operand<2, false, false>(matrix_fragment_8));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_149), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_145 = result_word;
    }

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_12 = permute_signed(matrix_fragment_5, matrix_fragment_5, 0x5432u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                      select_half_operand<2, false, false>(0u));

    packed_half_136 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5432u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5432u);

    packed_half_151 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(matrix_fragment_12));

    packed_half_136 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x4100410u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(packed_half_151));

    matrix_accumulator_134 = half_to_float_bits((matrix_fragment_10 & 65535u) * 65537u);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_145 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                     select_half_operand<2, false, false>(packed_half_145));

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_10 >> 16) * 65537u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_138 = half_to_float_bits((packed_half_136 & 65535u) * 65537u);

    matrix_accumulator_139 = half_to_float_bits((packed_half_136 >> 16) * 65537u);

    packed_half_136 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x4100410u);

    matrix_accumulator_137 = rsqrt_bits(matrix_fragment_10);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = permute_signed(packed_half_145, packed_half_145, 0x5432u);

    matrix_fragment_83 = half_to_float_bits((packed_half_136 & 65535u) * 65537u);

    matrix_fragment_82 = rsqrt_bits(matrix_accumulator_138);

    packed_half_136 = half_to_float_bits((packed_half_136 >> 16) * 65537u);

    packed_half_145 = add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                     select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_139 = rsqrt_bits(matrix_accumulator_139);

    packed_half_145 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_145), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_83 = rsqrt_bits(matrix_fragment_83);

    matrix_fragment_84 = half_to_float_bits((packed_half_145 & 65535u) * 65537u);

    packed_half_136 = rsqrt_bits(packed_half_136);

    packed_half_145 = half_to_float_bits((packed_half_145 >> 16) * 65537u);

    matrix_accumulator_139 = pack_half(matrix_accumulator_139, matrix_fragment_82);

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_134);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             packed_half_2, packed_half_3, matrix_fragment_78, matrix_fragment_79);

    matrix_fragment_154 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_154),
                                        select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_158 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_158),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_84 = rsqrt_bits(matrix_fragment_84);

    packed_half_136 = pack_half(packed_half_136, matrix_fragment_83);

    packed_half_145 = rsqrt_bits(packed_half_145);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_fragment_154 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(matrix_fragment_154));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_124),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_155 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_155),
                      select_half_operand<2, false, false>(matrix_accumulator_158));

    matrix_accumulator_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_136));

    matrix_accumulator_143 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_134 = pack_half(matrix_accumulator_137, matrix_accumulator_134);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_136));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(packed_half_136));

    packed_half_124 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    packed_half_145 = pack_half(packed_half_145, matrix_fragment_84);

    matrix_fragment_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                      select_half_operand<2, false, false>(matrix_accumulator_135));

    multiply_accumulate_half(packed_half_86, matrix_fragment_87, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_fragment_154, matrix_fragment_155, packed_half_86,
                             matrix_fragment_87);

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                       select_half_operand<2, false, false>(packed_half_124));

    matrix_fragment_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                      select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<0, false, false>(packed_half_144));

    packed_half_124 = 0x29c0u;

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(packed_half_144));

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(packed_half_144));

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_112,
                             matrix_fragment_113, matrix_accumulator_114, matrix_accumulator_115,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                      select_half_operand<2, false, false>(matrix_fragment_113));

    matrix_fragment_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                      select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(matrix_fragment_112));

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x3c203c20u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_81 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_9),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_23 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3c203c20u);

    matrix_fragment_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_11),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_98,
                             matrix_accumulator_99);

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_10),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    packed_half_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_86),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x3e473e47u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_fragment_23 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x3c203c20u);

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x3c203c20u);

    packed_half_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_86), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             packed_half_2, packed_half_3, matrix_fragment_94, matrix_fragment_95);

    matrix_fragment_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_accumulator_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x3e473e47u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_accumulator_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_86), 0x3e473e47u);

    matrix_fragment_23 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_79 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_22 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 = (matrix_fragment_9 << 0x5u) + 0x7ff88000u;

    matrix_fragment_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_78 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(packed_half_86, matrix_fragment_87, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_82, matrix_fragment_83, packed_half_104,
                             packed_half_105);

    matrix_fragment_80 = (matrix_fragment_10 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = (matrix_fragment_11 << 0x5u) + 0x7ff88000u;

    matrix_fragment_10 = (matrix_accumulator_109 << 0x5u) + 0x7ff88000u;

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                     select_half_operand<2, false, false>(matrix_fragment_79));

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                     select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_11 = (matrix_fragment_112 << 0x5u) + 0x7ff88000u;

    matrix_fragment_112 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                         select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                     select_half_operand<2, false, false>(packed_half_105));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                         select_half_operand<2, false, false>(matrix_fragment_112));

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_81 = (matrix_fragment_81 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_98),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_154, matrix_fragment_155, matrix_accumulator_102,
                             matrix_accumulator_103);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_9 = (matrix_accumulator_108 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_84, matrix_fragment_85, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_accumulator_108 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                       select_half_operand<2, false, false>(matrix_fragment_81));

    packed_half_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3e473e47u);

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                       select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_24 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_98), 0x3c203c20u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    matrix_accumulator_99 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99), 0x3c203c20u);

    matrix_accumulator_108 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                       select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                     select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    matrix_accumulator_109 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_24), 0x3e473e47u);

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_accumulator_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    matrix_accumulator_99 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99), 0x3e473e47u);

    matrix_accumulator_25 = (packed_half_104 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_24 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                       select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_accumulator_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_fragment_90 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                       select_half_operand<2, false, false>(matrix_accumulator_24));

    packed_half_105 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_108 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    packed_half_104 = (matrix_accumulator_98 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_106 = (matrix_accumulator_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_112 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_112),
                                         select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_96 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_97 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_107 = (matrix_accumulator_107 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_109 = (matrix_accumulator_109 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                       select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_94 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_fragment_113 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                         select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_102 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3c203c20u);

    matrix_fragment_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    packed_half_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_86),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_26),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3c203c20u);

    matrix_accumulator_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_27),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_fragment_113 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                         select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_92 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3e473e47u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    packed_half_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_86), 0x3c203c20u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_accumulator_26 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3e473e47u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_accumulator_27 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_27), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    packed_half_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_86), 0x3e473e47u);

    matrix_fragment_92 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_accumulator_114 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_27), 0x3e473e47u);

    matrix_fragment_94 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_64, matrix_fragment_65, 0u, 0u);

    matrix_accumulator_27 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_accumulator_26 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_26 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = (packed_half_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_93 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    packed_half_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                    select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_64 = (matrix_accumulator_114 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_113 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                                         select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(packed_half_86));

    matrix_fragment_88 = address_word_161 * 0x4u;

    matrix_accumulator_100 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_89 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    packed_half_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                    select_half_operand<2, false, false>(matrix_fragment_64));

    condition_1 = (address_word_161 & 0x1u) != 0;

    matrix_fragment_113 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_113),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    packed_half_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                    select_half_operand<2, false, false>(packed_half_86));

    matrix_accumulator_114 = matrix_fragment_88 & 0x1cu;

    matrix_fragment_88 = condition_1 ? matrix_fragment_87 : matrix_fragment_112;

    matrix_fragment_87 = condition_1 ? matrix_fragment_112 : matrix_fragment_87;

    condition_0 = (address_word_161 & 0x2u) != 0;

    matrix_accumulator_115 = condition_1 ? packed_half_86 : matrix_fragment_113;

    matrix_fragment_112 = condition_1 ? matrix_fragment_113 : packed_half_86;

    matrix_fragment_89 = uint32_t(int64_t(int32_t(address_word_161)) * (1ll << 0x1du) >> 32) +
                         matrix_accumulator_114;

    matrix_accumulator_114 = !condition_0 ? matrix_fragment_88 : matrix_accumulator_115;

    matrix_accumulator_126 = !condition_0 ? matrix_accumulator_115 : matrix_fragment_88;

    matrix_fragment_113 = !condition_0 ? matrix_fragment_87 : matrix_fragment_112;

    packed_half_86 = 0x1u ^ matrix_fragment_89;

    matrix_accumulator_115 = !condition_0 ? matrix_fragment_112 : matrix_fragment_87;

    matrix_fragment_87 = 0x2u ^ matrix_fragment_89;

    matrix_fragment_88 = 0x3u ^ matrix_fragment_89;

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(matrix_fragment_89), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_113), "r"(packed_half_86), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_113 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_126), "r"(matrix_fragment_87), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_126 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_88), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    condition_0 = (address_word_161 & 0x8u) != 0;

    condition_1 = (address_word_161 & 0x10u) != 0;

    matrix_fragment_112 = !condition_0 ? matrix_accumulator_114 : matrix_fragment_113;

    matrix_accumulator_125 = !condition_0 ? matrix_fragment_113 : matrix_accumulator_114;

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_132,
                             matrix_accumulator_133);

    matrix_accumulator_127 = !condition_0 ? matrix_accumulator_126 : matrix_accumulator_115;

    matrix_accumulator_114 = !condition_0 ? matrix_accumulator_115 : matrix_accumulator_126;

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_52 = !condition_1 ? matrix_fragment_112 : matrix_accumulator_127;

    matrix_fragment_53 = !condition_1 ? matrix_accumulator_125 : matrix_accumulator_114;

    matrix_fragment_112 = !condition_1 ? matrix_accumulator_127 : matrix_fragment_112;

    matrix_accumulator_114 = !condition_1 ? matrix_accumulator_114 : matrix_accumulator_125;

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_112));

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_52 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_52),
                                        select_half_operand<1, false, false>(matrix_fragment_52));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_52, 0x5410u);

    packed_half_147 = address_word_161;

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x4100410u);

    matrix_fragment_112 = uint32_t(int32_t(packed_half_147) >> 0x2u);

    matrix_fragment_53 = half_to_float_bits((matrix_fragment_52 & 65535u) * 65537u);

    matrix_fragment_52 = half_to_float_bits((matrix_fragment_52 >> 16) * 65537u);

    matrix_fragment_113 = matrix_fragment_112 + 0x8u;

    matrix_fragment_53 = reciprocal_bits(matrix_fragment_53);

    matrix_accumulator_115 = uint32_t(int32_t(matrix_fragment_113) >> 0x1fu);

    matrix_fragment_52 = reciprocal_bits(matrix_fragment_52);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_6,
                             matrix_fragment_7);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_54 =
        uint32_t(combine_words(matrix_accumulator_115, 0u) << 0x5u >> 32) + matrix_fragment_113;

    matrix_fragment_54 = matrix_fragment_54 & 0xffffffe0u;

    matrix_fragment_55 = uint32_t(int32_t(packed_half_147) >> 0x1fu);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_98,
                             matrix_accumulator_99);

    matrix_accumulator_125 = pack_half(matrix_fragment_52, matrix_fragment_53);

    matrix_fragment_55 =
        uint32_t(combine_words(matrix_fragment_55, 0u) << 0x5u >> 32) + matrix_fragment_112;

    matrix_fragment_55 = matrix_fragment_55 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_fragment_60 = matrix_fragment_113 + (0u - matrix_fragment_54);

    matrix_fragment_17 = transpose_half_tile(matrix_fragment_17);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_fragment_60), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_60 = result_word;
    }

    matrix_accumulator_114 = matrix_fragment_112 + (0u - matrix_fragment_55);

    matrix_fragment_16 = transpose_half_tile(matrix_fragment_16);

    matrix_accumulator_117 = transpose_half_tile(matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_accumulator_114), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_54 = result_word;
    }

    matrix_fragment_61 = transpose_half_tile(matrix_accumulator_116);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    matrix_accumulator_116 = permute_signed(matrix_fragment_60, matrix_fragment_60, 0x5410u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = permute_signed(matrix_fragment_54, matrix_fragment_54, 0x5410u);

    matrix_fragment_17 = uint32_t(int64_t(int32_t(packed_half_147)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_16 = uint32_t(int32_t(matrix_fragment_17) >> 0x1fu);

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_76 = uint32_t(int64_t(int32_t(packed_half_147)) * (1ll << 0x1eu) >> 32) + 0x18u;

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_16 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x5u >> 32) + matrix_fragment_17;

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_80));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_fragment_63 = uint32_t(int32_t(matrix_fragment_76) >> 0x1fu);

    matrix_fragment_62 = matrix_fragment_16 & 0xffffffe0u;

    matrix_fragment_63 =
        uint32_t(combine_words(matrix_fragment_63, 0u) << 0x5u >> 32) + matrix_fragment_76;

    matrix_accumulator_111 = transpose_half_tile(matrix_accumulator_111);

    matrix_fragment_80 = matrix_fragment_17 + (0u - matrix_fragment_62);

    matrix_fragment_17 = matrix_fragment_63 & 0xffffffe0u;

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_accumulator_110 = transpose_half_tile(matrix_accumulator_110);

    matrix_accumulator_126 = matrix_fragment_76 + (0u - matrix_fragment_17);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_fragment_80), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_80 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_125), "r"(matrix_accumulator_126), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_126 = result_word;
    }

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_19);

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_18);

    matrix_fragment_16 = transpose_half_tile(matrix_fragment_21);

    matrix_fragment_121 = transpose_half_tile(matrix_fragment_121);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 = transpose_half_tile(matrix_accumulator_119);

    matrix_accumulator_111 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = transpose_half_tile(matrix_fragment_7);

    matrix_fragment_63 = transpose_half_tile(matrix_fragment_120);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_118 = transpose_half_tile(matrix_accumulator_118);

    matrix_fragment_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                      select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_62 = transpose_half_tile(matrix_fragment_20);

    matrix_fragment_6 = transpose_half_tile(matrix_fragment_6);

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_81 = transpose_half_tile(matrix_fragment_59);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_110 = permute_signed(matrix_fragment_80, matrix_fragment_80, 0x5410u);

    matrix_fragment_80 = transpose_half_tile(matrix_fragment_58);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 =
        permute_signed(matrix_accumulator_126, matrix_accumulator_126, 0x5410u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_121),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_accumulator_119 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                      select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                      select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_106 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_133 = transpose_half_tile(matrix_accumulator_133);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = transpose_half_tile(matrix_fragment_57);

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_99 = transpose_half_tile(matrix_accumulator_99);

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 = transpose_half_tile(matrix_accumulator_103);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_101 = transpose_half_tile(matrix_accumulator_101);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_67 = transpose_half_tile(matrix_accumulator_67);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                      select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_132 = transpose_half_tile(matrix_accumulator_132);

    matrix_fragment_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                      select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_56 = transpose_half_tile(matrix_fragment_56);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_80 = transpose_half_tile(matrix_accumulator_98);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_81 = transpose_half_tile(matrix_accumulator_102);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_96 = transpose_half_tile(matrix_accumulator_100);

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_66 = transpose_half_tile(matrix_accumulator_66);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_106));

    packed_half_146 = threadIdx.y;

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_accumulator_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_67 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                      select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_56 = packed_half_147 & 0x7ffffffcu;

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_96, matrix_fragment_97, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_26));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_accumulator_102 = packed_half_146 * 0x20u;

    matrix_fragment_57 = 0u - matrix_fragment_56 + address_word_161;

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, packed_half_104,
                             packed_half_105);

    matrix_accumulator_100 = uint32_t(uint32_t(matrix_accumulator_102) >> 0x1u);

    matrix_fragment_57 = matrix_fragment_57 * 0x2u + matrix_accumulator_102;

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_96, matrix_fragment_97, matrix_accumulator_24,
                             matrix_accumulator_25);

    matrix_accumulator_98 = matrix_accumulator_100 + 0x4u;

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_57) >> 0x1u);

    matrix_fragment_11 = matrix_accumulator_98 + address_word_161 + (0u - index_word_163);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_99 = matrix_accumulator_100 + 0x8u;

    matrix_accumulator_100 = matrix_accumulator_100 + 0xcu;

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_107 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_accumulator_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = matrix_accumulator_99 + address_word_161 + (0u - index_word_163);

    matrix_fragment_8 = (packed_half_146 << 0x9u) + uniform_index_word_4;

    matrix_fragment_65 = matrix_accumulator_100 + address_word_161 + (0u - index_word_163);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_101 = address_word_161 * 0x10u + matrix_fragment_8;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_77));

    split_address_words(matrix_fragment_56, matrix_fragment_57,
                        uint64_t(uint32_t(matrix_fragment_57)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

    split_address_words(matrix_fragment_64, matrix_fragment_65,
                        uint64_t(uint32_t(matrix_fragment_65)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_accumulator_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                          select_half_operand<2, false, false>(matrix_fragment_11));

    packed_half_104 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_56, matrix_fragment_57) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(matrix_fragment_52));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_101, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_101, matrix_fragment_16,
                                         matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                         shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_101 + 0x400u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    packed_half_105 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_64, matrix_fragment_65) + 0xf0b0ull, memory_ranges,
        error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_101 + 0x400u, matrix_accumulator_24, matrix_accumulator_25,
        matrix_accumulator_26, matrix_accumulator_27, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    matrix_fragment_91 = packed_half_146 * 0x100u;

    split_address_words(matrix_fragment_90, matrix_fragment_91,
                        uint64_t(uint32_t(matrix_fragment_91)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_164, address_word_165));

    split_address_words(matrix_fragment_90, matrix_fragment_91,
                        int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_90, matrix_fragment_91));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe0b0ull, matrix_accumulator_24,
        matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe2b0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe8b0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xeab0ull, matrix_fragment_64,
        matrix_fragment_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0, matrix_fragment_76,
                                        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_92 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_93 = unpack_fp8_bits(matrix_fragment_9);

    matrix_accumulator_108 = unpack_fp8_bits(matrix_fragment_11);

    uniform_index_word_4 = blockIdx.y;

    matrix_accumulator_103 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    uniform_block_index_5 = blockIdx.x;

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_parameter_word_7;

    uniform_block_index_5 = (uniform_block_index_5 << 0x3u) + uniform_index_word_6;

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_accumulator_108 = parameters[parameter_optional_control_20];
    matrix_accumulator_109 = parameters[parameter_optional_control_21];

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(packed_half_105));

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x200u, matrix_fragment_92,
                                        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                        shared_bytes, memory_ranges, error_counts);

    condition_2 = int32_t(matrix_accumulator_108) > int32_t(0u);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                             matrix_accumulator_117);

    condition_3 = int32_t(matrix_accumulator_109) > int32_t(0u);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_8,
                             matrix_fragment_9);

    index_word_156 = condition_2 ? matrix_accumulator_108 : index_word_156;

    index_word_157 = condition_3 ? matrix_accumulator_109 : index_word_157;

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_10 =
        uint32_t(combine_words(matrix_fragment_112, 0u) << 0x2u >> 32) + matrix_fragment_112;

    matrix_accumulator_103 = matrix_fragment_10 & 0xfffffffcu;

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_accumulator_111 = address_word_161;

    matrix_accumulator_110 =
        matrix_fragment_112 + uniform_block_index_5 + (0u - matrix_accumulator_103);

    matrix_accumulator_111 =
        uint32_t(int64_t(int32_t(matrix_accumulator_111)) * (1ll << 0x1cu) >> 32) +
        uniform_index_word_4;

    condition_3 = int32_t(matrix_accumulator_110) >= int32_t(index_word_157);

    condition_2 = int32_t(matrix_accumulator_111) >= int32_t(index_word_156);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_116,
                             matrix_accumulator_117);

    condition_3 = int32_t(matrix_accumulator_110) < int32_t(0u) || condition_3;

    condition_2 = int32_t(matrix_accumulator_111) < int32_t(0u) || condition_2;

    matrix_accumulator_117 = matrix_accumulator_111 + 0x2u;

    condition_5 = condition_2 | condition_3;

    condition_4 = int32_t(matrix_accumulator_117) >= int32_t(index_word_156);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9);

    condition_4 = int32_t(matrix_accumulator_117) < int32_t(0u) || condition_4;

    if (!condition_5) {
        matrix_accumulator_118 = parameters[parameter_output_low_2];
        matrix_accumulator_119 = parameters[parameter_output_high_3];
    }

    condition_6 = condition_4 | condition_3;

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_accumulator_103 = packed_half_146 * 0x2u;

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_accumulator_125 = matrix_accumulator_110 * 0x4u;
    }

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_accumulator_116 = index_word_156 * matrix_accumulator_103 + matrix_accumulator_111;
    }

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_fragment_120 = matrix_accumulator_110 * 0x4u;

        matrix_accumulator_108 = parameters[parameter_output_low_2];
        matrix_accumulator_109 = parameters[parameter_output_high_3];
    }

    matrix_fragment_92 = index_word_157 * 0x4u;

    if (!condition_6) {
        matrix_accumulator_127 = index_word_156 * matrix_accumulator_103 + matrix_accumulator_117;
    }

    matrix_fragment_121 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_accumulator_116 =
            matrix_fragment_92 * matrix_accumulator_116 + matrix_accumulator_125;
    }

    if (!condition_6) {
        matrix_fragment_8 = matrix_fragment_92 * matrix_accumulator_127 + matrix_fragment_120;
    }

    if (!condition_5) {
        matrix_accumulator_135 = permute_signed(matrix_fragment_76, matrix_fragment_121, 0x5410u);

        matrix_fragment_94 = parameters[parameter_output_low_2];
        matrix_fragment_95 = parameters[parameter_output_high_3];

        matrix_accumulator_125 = matrix_accumulator_116 + address_word_161 + (0u - index_word_163);
    }

    if (!condition_6) {
        matrix_accumulator_127 = matrix_fragment_8 + address_word_161 + (0u - index_word_163);
    }

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        split_address_words(matrix_accumulator_118, matrix_accumulator_119,
                            int64_t(int32_t(matrix_accumulator_125)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_118, matrix_accumulator_119));
    }

    matrix_fragment_121 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_accumulator_125 = permute_signed(matrix_fragment_10, matrix_fragment_121, 0x5410u);
    }

    if (!condition_6) {
        matrix_fragment_121 = permute_signed(matrix_fragment_77, matrix_fragment_8, 0x5410u);

        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    matrix_fragment_93 = matrix_accumulator_103 + 0x1u;

    if (!condition_6) {
        matrix_fragment_10 = matrix_accumulator_110 * 0x4u;

        matrix_fragment_77 = index_word_156 * matrix_fragment_93 + matrix_accumulator_117;

        matrix_fragment_10 = matrix_fragment_92 * matrix_fragment_77 + matrix_fragment_10;
    }

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_6) {
        matrix_fragment_77 = matrix_fragment_10 + address_word_161 + (0u - index_word_163);
    }

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_5) {
        matrix_accumulator_126 = matrix_accumulator_110 * 0x4u;
    }

    if (!condition_6) {
        matrix_fragment_79 = permute_signed(matrix_fragment_11, matrix_fragment_10, 0x5410u);

        split_address_words(matrix_fragment_76, matrix_fragment_77,
                            int64_t(int32_t(matrix_fragment_77)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x400u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    if (!condition_5) {
        matrix_accumulator_133 = index_word_156 * matrix_fragment_93 + matrix_accumulator_111;

        matrix_fragment_120 = matrix_fragment_92 * matrix_accumulator_133 + matrix_accumulator_126;

        matrix_accumulator_133 = matrix_fragment_120 + address_word_161 + (0u - index_word_163);
    }

    if (!condition_6) {
        split_address_words(matrix_accumulator_108, matrix_accumulator_109,
                            int64_t(int32_t(matrix_accumulator_127)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_108, matrix_accumulator_109));
    }

    if (!condition_5) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_118, matrix_accumulator_119),
                                    matrix_accumulator_135, memory_ranges, error_counts);

        split_address_words(matrix_fragment_94, matrix_fragment_95,
                            int64_t(int32_t(matrix_accumulator_133)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_94, matrix_fragment_95));
    }

    if (!condition_6) {
        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_108, matrix_accumulator_109),
                                    matrix_fragment_121, memory_ranges, error_counts);
    }

    matrix_fragment_78 = unpack_fp8_bits(matrix_fragment_17);

    if (!condition_5) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_94, matrix_fragment_95),
                                    matrix_accumulator_125, memory_ranges, error_counts);
    }

    if (!condition_6) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_76, matrix_fragment_77),
                                    matrix_fragment_79, memory_ranges, error_counts);
    }

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_77 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x600u, matrix_fragment_76,
                                        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(packed_half_104));

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(packed_half_105));

    packed_half_0 = matrix_accumulator_110 + 0x4u;

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17);

    condition_5 = int32_t(packed_half_0) >= int32_t(index_word_157);

    condition_5 = int32_t(packed_half_0) < int32_t(0u) || condition_5;

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_18,
                             matrix_fragment_19);

    condition_2 = condition_2 | condition_5;

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_26,
                             matrix_accumulator_27);

    condition_4 = condition_4 | condition_5;

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_24,
                             matrix_accumulator_25);

    if (!condition_2) {
        matrix_fragment_54 = index_word_156 * matrix_fragment_93 + matrix_accumulator_111;
    }

    if (!condition_4) {
        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_53 = packed_half_0 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_16,
                             matrix_fragment_17);

    if (!condition_2) {
        matrix_fragment_16 = parameters[parameter_output_low_2];
        matrix_fragment_17 = parameters[parameter_output_high_3];

        matrix_fragment_57 = packed_half_0 * 0x4u;

        matrix_fragment_54 = matrix_fragment_92 * matrix_fragment_54 + matrix_fragment_57;

        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_18,
                             matrix_fragment_19);

    if (!condition_4) {
        matrix_fragment_56 = parameters[parameter_output_low_2];
        matrix_fragment_57 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_18 = index_word_156 * matrix_accumulator_103 + matrix_accumulator_111;
    }

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_accumulator_26 = matrix_fragment_92 * matrix_fragment_18 + matrix_fragment_53;
    }

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_accumulator_27 = matrix_accumulator_26 + address_word_161 + (0u - index_word_163);
    }

    if (!condition_4) {
        matrix_fragment_55 = packed_half_0 * 0x4u;

        matrix_accumulator_25 = permute_signed(matrix_accumulator_25, matrix_fragment_18, 0x5410u);

        matrix_fragment_52 = index_word_156 * matrix_accumulator_103 + matrix_accumulator_117;
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (!condition_4) {
        matrix_accumulator_27 = packed_half_0 * 0x4u;

        matrix_fragment_18 = index_word_156 * matrix_fragment_93 + matrix_accumulator_117;

        matrix_fragment_52 = matrix_fragment_92 * matrix_fragment_52 + matrix_fragment_55;
    }

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        matrix_fragment_18 = matrix_fragment_92 * matrix_fragment_18 + matrix_accumulator_27;
    }

    if (!condition_2) {
        matrix_fragment_55 = matrix_fragment_54 + address_word_161 + (0u - index_word_163);
    }

    if (!condition_4) {
        matrix_fragment_53 = matrix_fragment_52 + address_word_161 + (0u - index_word_163);
    }

    if (!condition_2) {
        matrix_fragment_59 = permute_signed(matrix_accumulator_24, matrix_fragment_19, 0x5410u);
    }

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        index_word_163 = matrix_fragment_18 + address_word_161 + (0u - index_word_163);
    }

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_4) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_53)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (!condition_2) {
        matrix_fragment_19 = permute_signed(matrix_fragment_64, matrix_fragment_19, 0x5410u);
    }

    if (!condition_4) {
        matrix_fragment_65 = permute_signed(matrix_fragment_65, matrix_fragment_18, 0x5410u);
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_55)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                    matrix_fragment_59, memory_ranges, error_counts);
    }

    if (!condition_4) {
        split_address_words(matrix_fragment_56, matrix_fragment_57,
                            int64_t(int32_t(index_word_163)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_56, matrix_fragment_57));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_accumulator_25, memory_ranges, error_counts);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_19, memory_ranges, error_counts);
    }

    if (!condition_4) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_56, matrix_fragment_57),
                                    matrix_fragment_65, memory_ranges, error_counts);
    }

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xb0a0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xb2a0ull, matrix_accumulator_24,
        matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xb4a0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xb6a0ull, matrix_fragment_16,
        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xb8a0ull, matrix_fragment_64,
        matrix_fragment_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xbaa0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_48,
                             matrix_fragment_49);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xbca0ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_122, matrix_fragment_123) + 0xbea0ull, matrix_fragment_120,
        matrix_fragment_121, matrix_fragment_122, matrix_fragment_123, memory_ranges, error_counts);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    packed_half_104 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                     select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                      select_half_operand<2, false, false>(matrix_fragment_77),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_104), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_105 = result_word;
    }

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_71 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    packed_half_105 = add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                     select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_105), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_44 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    matrix_fragment_44 = add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                        select_half_operand<2, false, false>(matrix_fragment_44));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_29 = permute_signed(matrix_fragment_44, matrix_fragment_44, 0x5432u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_44 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_29 = permute_signed(matrix_fragment_40, matrix_fragment_40, 0x5432u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x4100410u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x4100410u);

    matrix_fragment_30 = half_to_float_bits((matrix_fragment_44 & 65535u) * 65537u);

    matrix_accumulator_39 = half_to_float_bits((matrix_fragment_40 & 65535u) * 65537u);

    matrix_fragment_40 = half_to_float_bits((matrix_fragment_40 >> 16) * 65537u);

    matrix_fragment_44 = half_to_float_bits((matrix_fragment_44 >> 16) * 65537u);

    matrix_fragment_30 = rsqrt_bits(matrix_fragment_30);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_accumulator_34));

    matrix_fragment_31 = rsqrt_bits(matrix_fragment_44);

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_accumulator_39 = rsqrt_bits(matrix_accumulator_39);

    matrix_fragment_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_40 = rsqrt_bits(matrix_fragment_40);

    matrix_accumulator_38 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_47 = pack_half(matrix_fragment_31, matrix_fragment_30);

    matrix_fragment_30 = pack_half(matrix_fragment_40, matrix_accumulator_39);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_28));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_38), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_39 = result_word;
    }

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_44 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_39));

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_8,
                             matrix_fragment_9);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_44), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_2, packed_half_3, matrix_fragment_10, matrix_fragment_11);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_10 = permute_signed(matrix_fragment_9, matrix_fragment_9, 0x5432u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_8 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5432u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x4100410u);

    matrix_fragment_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x4100410u);

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_9 & 65535u) * 65537u);

    matrix_fragment_9 = half_to_float_bits((matrix_fragment_9 >> 16) * 65537u);

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_11 & 65535u) * 65537u);

    matrix_fragment_11 = half_to_float_bits((matrix_fragment_11 >> 16) * 65537u);

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    matrix_fragment_9 = rsqrt_bits(matrix_fragment_9);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    matrix_fragment_11 = rsqrt_bits(matrix_fragment_11);

    matrix_fragment_10 = pack_half(matrix_fragment_9, matrix_fragment_10);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_44 = pack_half(matrix_fragment_11, matrix_fragment_8);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(packed_half_144));

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_24,
                             matrix_accumulator_25);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<0, false, false>(packed_half_144));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_154, matrix_fragment_155, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_16,
                             matrix_fragment_17);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_64,
                             matrix_fragment_65);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_2, packed_half_3, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_56,
                             matrix_fragment_57);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_accumulator_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_24),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_26),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_accumulator_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_25),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_38 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3c203c20u);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_27),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_accumulator_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_154, matrix_fragment_155, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_24 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_24), 0x3c203c20u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    packed_half_2 = maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       0x3c203c20u);

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_38 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_38), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_82, matrix_fragment_83, matrix_fragment_120,
                             matrix_fragment_121);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    matrix_accumulator_25 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_25), 0x3c203c20u);

    matrix_accumulator_26 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_24), 0x3e473e47u);

    packed_half_2 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0x3e473e47u);

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_122, matrix_fragment_123, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_122,
                             matrix_fragment_123);

    matrix_accumulator_39 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x3c203c20u);

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_accumulator_27 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_25), 0x3e473e47u);

    packed_half_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x3e473e47u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_39 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_39), 0x3e473e47u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_12 = (matrix_accumulator_38 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_25 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_26 = (matrix_accumulator_26 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = (packed_half_2 << 0x5u) + 0x7ff88000u;

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_28 = (packed_half_3 << 0x5u) + 0x7ff88000u;

    matrix_fragment_30 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                       select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_fragment_13 = (matrix_accumulator_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_24 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_27 = (matrix_accumulator_27 << 0x5u) + 0x7ff88000u;

    packed_half_2 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    packed_half_3 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3c203c20u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3c203c20u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_accumulator_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                       select_half_operand<2, false, false>(matrix_accumulator_24));

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3e473e47u);

    matrix_accumulator_33 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3e473e47u);

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3c203c20u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    matrix_fragment_41 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_32 = (matrix_fragment_55 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = (matrix_fragment_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_17),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_30 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_18),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_accumulator_33 = (matrix_fragment_54 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_36 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_38 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_39 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_120),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3e473e47u);

    matrix_fragment_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_121),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    matrix_fragment_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_122),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x3c203c20u);

    matrix_fragment_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_123),
        select_half_operand<0, false, false>(packed_half_124), 0x3d343d34u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3c203c20u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3c203c20u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x3e473e47u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_accumulator_34 = (matrix_fragment_48 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_35 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                       select_half_operand<2, false, false>(matrix_accumulator_39));

    matrix_fragment_17 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x3c203c20u);

    matrix_fragment_18 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0x3c203c20u);

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0x3c203c20u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3e473e47u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3e473e47u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120), 0x3c203c20u);

    matrix_fragment_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121), 0x3c203c20u);

    matrix_fragment_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122), 0x3c203c20u);

    matrix_fragment_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123), 0x3c203c20u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x3e473e47u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0x3e473e47u);

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                       select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0x3e473e47u);

    matrix_fragment_43 = (matrix_fragment_16 << 0x5u) + 0x7ff88000u;

    matrix_fragment_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120), 0x3e473e47u);

    matrix_fragment_121 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121), 0x3e473e47u);

    matrix_fragment_122 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122), 0x3e473e47u);

    matrix_fragment_123 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123), 0x3e473e47u);

    matrix_fragment_16 = (matrix_fragment_49 << 0x5u) + 0x7ff88000u;

    matrix_fragment_17 = (matrix_fragment_51 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_44 = (matrix_fragment_18 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = (matrix_fragment_19 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_40 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = (matrix_fragment_120 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = (matrix_fragment_121 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_fragment_122 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_fragment_123 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    condition_2 = (address_word_161 & 0x1u) != 0;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_9 = condition_2 ? matrix_fragment_11 : matrix_fragment_8;

    condition_6 = (address_word_161 & 0x2u) != 0;

    matrix_fragment_8 = condition_2 ? matrix_fragment_8 : matrix_fragment_11;

    matrix_fragment_10 = condition_2 ? matrix_fragment_49 : matrix_fragment_46;

    matrix_fragment_49 = condition_2 ? matrix_fragment_46 : matrix_fragment_49;

    matrix_fragment_46 = !condition_6 ? matrix_fragment_9 : matrix_fragment_10;

    matrix_fragment_11 = !condition_6 ? matrix_fragment_8 : matrix_fragment_49;

    matrix_fragment_10 = !condition_6 ? matrix_fragment_10 : matrix_fragment_9;

    matrix_fragment_49 = !condition_6 ? matrix_fragment_49 : matrix_fragment_8;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_89), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(packed_half_86), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_87), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(matrix_fragment_88), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    matrix_fragment_8 = !condition_0 ? matrix_fragment_46 : matrix_fragment_11;

    matrix_fragment_9 = !condition_0 ? matrix_fragment_11 : matrix_fragment_46;

    matrix_fragment_47 = !condition_0 ? matrix_fragment_10 : matrix_fragment_49;

    matrix_fragment_48 = !condition_0 ? matrix_fragment_49 : matrix_fragment_10;

    matrix_fragment_50 = !condition_1 ? matrix_fragment_8 : matrix_fragment_47;

    matrix_fragment_51 = !condition_1 ? matrix_fragment_9 : matrix_fragment_48;

    matrix_fragment_8 = !condition_1 ? matrix_fragment_47 : matrix_fragment_8;

    matrix_fragment_9 = !condition_1 ? matrix_fragment_48 : matrix_fragment_9;

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_50 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_50),
                                        select_half_operand<1, false, false>(matrix_fragment_50));

    matrix_fragment_50 = permute_signed(matrix_fragment_50, matrix_fragment_50, 0x5410u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x4100410u);

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_50 & 65535u) * 65537u);

    matrix_fragment_50 = half_to_float_bits((matrix_fragment_50 >> 16) * 65537u);

    matrix_fragment_8 = reciprocal_bits(matrix_fragment_8);

    matrix_fragment_9 = reciprocal_bits(matrix_fragment_50);

    matrix_accumulator_115 =
        uint32_t(combine_words(matrix_accumulator_115, 0u) << 0x5u >> 32) + matrix_fragment_113;

    matrix_fragment_46 = uint32_t(int64_t(int32_t(packed_half_147)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_10 = matrix_accumulator_115 & 0xffffffe0u;

    matrix_fragment_11 = uint32_t(int64_t(int32_t(packed_half_147)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_47 = uint32_t(int32_t(matrix_fragment_46) >> 0x1fu);

    matrix_fragment_10 = matrix_fragment_113 + (0u - matrix_fragment_10);

    matrix_fragment_47 =
        uint32_t(combine_words(matrix_fragment_47, 0u) << 0x5u >> 32) + matrix_fragment_46;

    matrix_fragment_9 = pack_half(matrix_fragment_9, matrix_fragment_8);

    matrix_fragment_8 = uint32_t(int32_t(matrix_fragment_11) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_114), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    matrix_fragment_8 =
        uint32_t(combine_words(matrix_fragment_8, 0u) << 0x5u >> 32) + matrix_fragment_11;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    matrix_fragment_47 = matrix_fragment_47 & 0xffffffe0u;

    matrix_fragment_8 = matrix_fragment_8 & 0xffffffe0u;

    matrix_fragment_48 = matrix_fragment_46 + (0u - matrix_fragment_47);

    matrix_fragment_50 = matrix_fragment_11 + (0u - matrix_fragment_8);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_48), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_50), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_accumulator_114 =
        permute_signed(matrix_accumulator_114, matrix_accumulator_114, 0x5410u);

    matrix_fragment_46 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5410u);

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_24 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_48 = permute_signed(matrix_fragment_48, matrix_fragment_48, 0x5410u);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_accumulator_24));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_47 = permute_signed(matrix_fragment_47, matrix_fragment_47, 0x5410u);

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_39),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_38),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_4 = parameters[parameter_trained_weights_low_4];
    matrix_fragment_5 = parameters[parameter_trained_weights_high_5];

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_35 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_24,
                             matrix_accumulator_25);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_96, matrix_fragment_97, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                      select_half_operand<2, false, false>(matrix_accumulator_35));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_101 + 0x800u, matrix_accumulator_32, matrix_accumulator_33,
        matrix_accumulator_34, matrix_accumulator_35, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    packed_half_2 = packed_half_147 & 0x7ffffffcu;

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_101 + 0xc00u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_60,
                             matrix_fragment_61);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_7 = 0u - packed_half_2 + address_word_161;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_96, matrix_fragment_97, matrix_fragment_62,
                             matrix_fragment_63);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_accumulator_102 = matrix_fragment_7 * 0x2u + matrix_accumulator_102;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = packed_half_147 & 0xfffffffcu;

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = matrix_accumulator_100 + address_word_161 + (0u - packed_half_2);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_13 = uint32_t(int32_t(matrix_accumulator_102) >> 0x1u);

    matrix_accumulator_25 = matrix_accumulator_98 + address_word_161 + (0u - packed_half_2);

    matrix_accumulator_27 = matrix_accumulator_99 + address_word_161 + (0u - packed_half_2);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                      select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_28, matrix_fragment_29,
                        uint64_t(uint32_t(matrix_fragment_29)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(packed_half_3));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_101, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_accumulator_24, matrix_accumulator_25,
                        uint64_t(uint32_t(matrix_accumulator_25)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    matrix_fragment_30 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                        uint64_t(uint32_t(matrix_accumulator_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_101 + 0x400u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_31 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_24, matrix_accumulator_25) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_36 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_26, matrix_accumulator_27) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_29 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_28, matrix_fragment_29) + 0xf0b0ull, memory_ranges,
        error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe0b0ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe2b0ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xe8b0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_90, matrix_fragment_91) + 0xeab0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    uniform_parameter_word_7 = 0u;

    uniform_index_word_6 = 0x400u;

    uniform_index_word_6 = (uniform_parameter_word_7 << 0x18u) + uniform_index_word_6;

    matrix_fragment_37 = (address_word_161 << 0x4u) + uniform_index_word_6;

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_37, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_3 = unpack_fp8_bits(matrix_accumulator_32);

    matrix_accumulator_24 = unpack_fp8_bits(matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_37 + 0x200u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_accumulator_32 >> 16);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_accumulator_33 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_accumulator_35);

    matrix_accumulator_35 = unpack_fp8_bits(matrix_accumulator_35 >> 16);

    uniform_index_word_4 = uniform_index_word_4 + 0x4u;

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    packed_half_3 = unpack_fp8_bits(matrix_accumulator_34);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_34 = unpack_fp8_bits(matrix_accumulator_34 >> 16);

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_26,
                             matrix_accumulator_27);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_37 + 0x400u, matrix_accumulator_24, matrix_accumulator_25,
        matrix_accumulator_26, matrix_accumulator_27, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_34),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_35),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_32,
                             matrix_accumulator_33);

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_3 = unpack_fp8_bits(matrix_fragment_16);

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_32,
                             matrix_accumulator_33);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    packed_half_3 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_34,
                             matrix_accumulator_35);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_3 = address_word_161;

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    packed_half_3 =
        uint32_t(int64_t(int32_t(packed_half_3)) * (1ll << 0x1cu) >> 32) + uniform_index_word_4;

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_37 + 0x600u, matrix_fragment_20, matrix_fragment_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    condition_0 = int32_t(packed_half_3) >= int32_t(index_word_156);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_30,
                             matrix_fragment_31);

    condition_0 = int32_t(packed_half_3) < int32_t(0u) || condition_0;

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_accumulator_24,
                             matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_7 = packed_half_3 + 0x2u;

    condition_1 = int32_t(matrix_fragment_7) >= int32_t(index_word_156);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_44,
                             matrix_fragment_45);

    condition_1 = int32_t(matrix_fragment_7) < int32_t(0u) || condition_1;

    condition_2 = condition_0 | condition_3;

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_fragment_112, 0u) << 0x2u >> 32) + matrix_fragment_112;

    condition_3 = condition_1 | condition_3;

    condition_0 = condition_0 | condition_5;

    multiply_accumulate_half(matrix_accumulator_38, matrix_accumulator_39, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_38,
                             matrix_accumulator_39);

    matrix_fragment_5 = matrix_fragment_4 & 0xfffffffcu;

    condition_1 = condition_1 | condition_5;

    matrix_fragment_112 = matrix_fragment_112 + uniform_block_index_5 + (0u - matrix_fragment_5);

    if (!condition_2) {
        matrix_fragment_5 = index_word_156 * matrix_accumulator_103 + packed_half_3;

        matrix_fragment_4 = matrix_fragment_112 * 0x4u;
    }

    if (!condition_3) {
        matrix_fragment_6 = matrix_fragment_112 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_46,
                             matrix_fragment_47);

    if (!condition_0) {
        matrix_fragment_17 = packed_half_0 * 0x4u;
    }

    if (!condition_1) {
        matrix_accumulator_25 = packed_half_0 * 0x4u;
    }

    if (!condition_2) {
        matrix_fragment_5 = matrix_fragment_92 * matrix_fragment_5 + matrix_fragment_4;
    }

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_32,
                             matrix_accumulator_33);

    if (!condition_2) {
        matrix_fragment_8 = matrix_fragment_112 * 0x4u;
    }

    if (!condition_3) {
        matrix_fragment_112 = matrix_fragment_112 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_40,
                             matrix_fragment_41);

    if (!condition_3) {
        matrix_fragment_9 = index_word_156 * matrix_accumulator_103 + matrix_fragment_7;
    }

    if (!condition_2) {
        matrix_fragment_11 = index_word_156 * matrix_fragment_93 + packed_half_3;
    }

    if (!condition_0) {
        matrix_fragment_10 = index_word_156 * matrix_accumulator_103 + packed_half_3;
    }

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_30,
                             matrix_fragment_31);

    if (!condition_1) {
        matrix_accumulator_103 = index_word_156 * matrix_accumulator_103 + matrix_fragment_7;
    }

    if (!condition_0) {
        matrix_fragment_12 = index_word_156 * matrix_fragment_93 + packed_half_3;
    }

    if (!condition_3) {
        packed_half_3 = index_word_156 * matrix_fragment_93 + matrix_fragment_7;
    }

    if (!condition_1) {
        matrix_fragment_93 = index_word_156 * matrix_fragment_93 + matrix_fragment_7;
    }

    if (!condition_3) {
        packed_half_3 = matrix_fragment_92 * packed_half_3 + matrix_fragment_112;
    }

    if (!condition_0) {
        matrix_fragment_7 = packed_half_0 * 0x4u;
    }

    if (!condition_1) {
        matrix_fragment_13 = packed_half_0 * 0x4u;
    }

    multiply_accumulate_half(matrix_accumulator_34, matrix_accumulator_35, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_34,
                             matrix_accumulator_35);

    if (!condition_3) {
        matrix_fragment_9 = matrix_fragment_92 * matrix_fragment_9 + matrix_fragment_6;

        matrix_fragment_37 = packed_half_3 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_2) {
        matrix_fragment_11 = matrix_fragment_92 * matrix_fragment_11 + matrix_fragment_8;
    }

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                  select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        packed_half_0 = matrix_fragment_92 * matrix_fragment_10 + matrix_fragment_7;
    }

    if (!condition_1) {
        matrix_fragment_4 = matrix_fragment_92 * matrix_accumulator_103 + matrix_fragment_13;
    }

    if (!condition_0) {
        matrix_fragment_6 = matrix_fragment_92 * matrix_fragment_12 + matrix_fragment_17;
    }

    matrix_accumulator_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_38),
                      select_half_operand<2, false, false>(0u));

    if (!condition_1) {
        matrix_fragment_92 = matrix_fragment_92 * matrix_fragment_93 + matrix_accumulator_25;
    }

    if (!condition_2) {
        matrix_accumulator_25 = matrix_fragment_5 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_1) {
        matrix_fragment_12 = parameters[parameter_output_low_2];
        matrix_fragment_13 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_29 = matrix_fragment_11 + address_word_161 + (0u - packed_half_2);
    }

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_18,
                             matrix_fragment_19);

    if (!condition_3) {
        matrix_accumulator_27 = matrix_fragment_9 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_0) {
        matrix_fragment_43 = packed_half_0 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_1) {
        matrix_fragment_49 = matrix_fragment_4 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_3) {
        matrix_fragment_8 = parameters[parameter_output_low_2];
        matrix_fragment_9 = parameters[parameter_output_high_3];
    }

    if (!condition_0) {
        matrix_fragment_51 = matrix_fragment_6 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_1) {
        address_word_161 = matrix_fragment_92 + address_word_161 + (0u - packed_half_2);
    }

    if (!condition_2) {
        matrix_fragment_53 = permute_signed(matrix_accumulator_38, packed_half_3, 0x5410u);

        packed_half_2 = parameters[parameter_output_low_2];
        packed_half_3 = parameters[parameter_output_high_3];
    }

    if (!condition_3) {
        matrix_fragment_4 = parameters[parameter_output_low_2];
        matrix_fragment_5 = parameters[parameter_output_high_3];
    }

    if (!condition_2) {
        matrix_fragment_6 = parameters[parameter_output_low_2];
        matrix_fragment_7 = parameters[parameter_output_high_3];
    }

    if (!condition_0) {
        matrix_fragment_10 = parameters[parameter_output_low_2];
        matrix_fragment_11 = parameters[parameter_output_high_3];

        matrix_fragment_14 = parameters[parameter_output_low_2];
        matrix_fragment_15 = parameters[parameter_output_high_3];
    }

    if (!condition_1) {
        matrix_fragment_16 = parameters[parameter_output_low_2];
        matrix_fragment_17 = parameters[parameter_output_high_3];
    }

    matrix_accumulator_39 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_39),
                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_34),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_accumulator_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_35),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_accumulator_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    matrix_accumulator_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    matrix_accumulator_33 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_3) {
        matrix_accumulator_39 = permute_signed(matrix_accumulator_39, packed_half_0, 0x5410u);
    }

    if (!condition_2) {
        matrix_accumulator_35 = permute_signed(matrix_fragment_46, matrix_fragment_21, 0x5410u);
    }

    if (!condition_3) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_37)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));
    }

    if (!condition_3) {
        matrix_fragment_47 = permute_signed(matrix_fragment_47, matrix_fragment_20, 0x5410u);
    }

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                  select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(packed_half_2, packed_half_3), matrix_fragment_53,
                                    memory_ranges, error_counts);
    }

    if (!condition_0) {
        matrix_fragment_23 = permute_signed(matrix_accumulator_32, matrix_fragment_23, 0x5410u);
    }

    if (!condition_1) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

        matrix_accumulator_33 = permute_signed(matrix_accumulator_33, matrix_fragment_22, 0x5410u);
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_accumulator_39, memory_ranges, error_counts);
    }

    if (!condition_0) {
        matrix_fragment_21 = permute_signed(matrix_fragment_30, matrix_fragment_21, 0x5410u);

        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_51)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_1) {
        matrix_fragment_31 = permute_signed(matrix_fragment_31, packed_half_0, 0x5410u);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_accumulator_35, memory_ranges, error_counts);
    }

    if (!condition_1) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(address_word_161)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (!condition_3) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_47, memory_ranges, error_counts);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_fragment_23, memory_ranges, error_counts);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_12, matrix_fragment_13),
                                    matrix_accumulator_33, memory_ranges, error_counts);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_14, matrix_fragment_15),
                                    matrix_fragment_21, memory_ranges, error_counts);
    }

    if (!condition_1) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                    matrix_fragment_31, memory_ranges, error_counts);
    }

    __syncthreads();
    return;
}
extern "C" __global__
__maxnreg__(160) void phoenix_window_attention_channels_64_output_view(KernelArguments parameters) {
    run_window_attention_channels_64_output_view<false>(parameters.words, nullptr, nullptr);
}
