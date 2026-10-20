// DLSS5 Phoenix optimized kernel: window_attention/channels_256/attention.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES (__CUDA_ARCH__ >= 900)
#define DLSS5_TILE_SHARED_BYTES 17408
#include "../../../common/device_memory_vector.cuh"
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
__device__ __forceinline__ void run_window_attention_channels_256_attention(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t matrix_accumulator_8 = 0;
    uint32_t matrix_accumulator_9 = 0;
    uint32_t matrix_fragment_10 = 0;
    uint32_t matrix_fragment_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t matrix_accumulator_16 = 0;
    uint32_t matrix_accumulator_17 = 0;
    uint32_t matrix_accumulator_18 = 0;
    uint32_t matrix_accumulator_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_accumulator_22 = 0;
    uint32_t matrix_accumulator_23 = 0;
    uint32_t matrix_accumulator_24 = 0;
    uint32_t matrix_accumulator_25 = 0;
    uint32_t matrix_accumulator_26 = 0;
    uint32_t matrix_accumulator_27 = 0;
    uint32_t matrix_accumulator_28 = 0;
    uint32_t matrix_accumulator_29 = 0;
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
    uint32_t matrix_accumulator_100 = 0;
    uint32_t matrix_accumulator_101 = 0;
    uint32_t matrix_accumulator_102 = 0;
    uint32_t matrix_accumulator_103 = 0;
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
    uint32_t address_word_142 = 0;
    uint32_t address_word_143 = 0;
    uint32_t tile_word_144 = 0;
    uint32_t index_word_145 = 0;
    uint32_t index_word_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t index_word_148 = 0;
    uint32_t index_word_149 = 0;
    uint32_t index_word_150 = 0;
    uint32_t index_word_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t index_word_155 = 0;
    uint32_t index_word_156 = 0;
    uint32_t index_word_157 = 0;
    uint32_t thread_row_158 = 0;
    uint32_t address_word_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t address_word_164 = 0;
    uint32_t address_word_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_parameter_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_reused_word_8 = 0;
    uint32_t uniform_reused_word_9 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool condition_6 = false;
    bool uniform_condition_0 = false;
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    uniform_parameter_word_6 = parameters[parameter_geometry_or_control_14];

    matrix_accumulator_8 = parameters[parameter_tensor_width_9];

    matrix_fragment_10 = blockIdx.y;

    uniform_parameter_word_7 = parameters[parameter_geometry_or_control_15];

    address_word_147 = 0u;

    matrix_fragment_7 = blockIdx.x;

    uniform_index_word_5 = parameters[parameter_tensor_height_8];

    matrix_fragment_4 = parameters[parameter_geometry_or_control_10];
    matrix_fragment_5 = parameters[parameter_geometry_or_control_11];

    matrix_fragment_11 = threadIdx.x;

    uniform_reused_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    matrix_accumulator_136 = threadIdx.y;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_6));

    matrix_fragment_6 = threadIdx.z;

    matrix_accumulator_2 = matrix_accumulator_8 + (0u - uniform_index_word_4);

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_7));

    uniform_index_word_4 = 0u - uniform_index_word_4 + uniform_index_word_5;

    condition_3 = uint32_t(matrix_accumulator_2) != 0u;

    matrix_accumulator_3 = min(int32_t(0u), int32_t(matrix_fragment_4));

    uniform_condition_0 = uint32_t(uniform_index_word_4) != 0u;

    index_word_0 = matrix_fragment_10 * 0x8u + matrix_fragment_5;

    matrix_fragment_4 = matrix_fragment_7 * 0x8u + matrix_fragment_4;

    matrix_fragment_5 = 0u - matrix_accumulator_3 + matrix_accumulator_8;

    matrix_fragment_7 = index_word_0 + (0u - uniform_parameter_word_7);

    matrix_accumulator_9 = matrix_fragment_4 + (0u - uniform_parameter_word_6);

    condition_4 = uint32_t(matrix_fragment_5) != 0u;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_4 + 0xffffffffu;
    }

    matrix_accumulator_3 = matrix_fragment_7 + 0x8u;

    matrix_accumulator_8 = matrix_fragment_7 + 0x7u;

    uniform_index_word_4 = 0u;

    if (uniform_condition_0) {
        uniform_parameter_word_6 = uint32_t(int32_t(uniform_index_word_5) >> 0x1fu);
    }

    matrix_fragment_7 = max(int32_t(0u), int32_t(matrix_fragment_7));

    condition_1 = int32_t(matrix_accumulator_3) > int32_t(0u);

    matrix_accumulator_8 = uint32_t(combine_words(matrix_accumulator_8, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_parameter_word_6 =
            uint32_t(combine_words(uniform_parameter_word_6, 0u) << 0x3u >> 32) +
            uniform_index_word_5;
    }

    if (condition_3) {
        matrix_accumulator_2 = matrix_accumulator_2 + 0xffffffffu;
    }

    condition_0 = (matrix_accumulator_136 | matrix_fragment_11 | matrix_fragment_6) != 0;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_6)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_8 = condition_1 ? matrix_accumulator_8 : 0u;

    matrix_fragment_7 = uint32_t(uint32_t(matrix_fragment_7) >> 0x3u);

    if (condition_3) {
        matrix_accumulator_3 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1fu);
    }

    matrix_fragment_12 = min(int32_t(matrix_accumulator_8), int32_t(uniform_index_word_4));

    if (condition_4) {
        matrix_fragment_5 = matrix_fragment_5 + 0xffffffffu;
    }

    matrix_fragment_6 = matrix_accumulator_9 + 0x8u;

    if (condition_3) {
        matrix_accumulator_3 =
            uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x3u >> 32) + matrix_accumulator_2;
    }

    condition_0 = int32_t(matrix_fragment_7) >= int32_t(matrix_fragment_12) || condition_0;

    if (condition_4) {
        matrix_accumulator_2 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);
    }

    condition_2 = int32_t(matrix_fragment_6) > int32_t(0u);

    matrix_fragment_6 = 0u;

    if (condition_3) {
        matrix_fragment_6 =
            uint32_t(int64_t(int32_t(matrix_accumulator_3)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_8 = matrix_accumulator_9 + 0x7u;

    if (condition_4) {
        matrix_accumulator_2 =
            uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x3u >> 32) + matrix_fragment_5;
    }

    matrix_accumulator_3 = uint32_t(int32_t(index_word_0) >> 0x1fu);

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    if (condition_4) {
        matrix_accumulator_2 = uint32_t(int32_t(matrix_accumulator_2) >> 0x3u);
    }

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + index_word_0;

    matrix_accumulator_8 = uint32_t(combine_words(matrix_accumulator_8, 0u) << 0x1du >> 32) + 0x1u;

    if (condition_4) {
        address_word_147 = matrix_accumulator_2 * matrix_fragment_10 + matrix_fragment_10;
    }

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_4;

    index_word_150 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    matrix_accumulator_9 = max(int32_t(0u), int32_t(matrix_accumulator_9));

    index_word_148 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    matrix_accumulator_3 = condition_2 ? matrix_accumulator_8 : 0u;
    if (!condition_0) {
        matrix_accumulator_9 = uint32_t(uint32_t(matrix_accumulator_9) >> 0x3u);

        matrix_accumulator_8 = min(int32_t(matrix_accumulator_3), int32_t(matrix_fragment_6));
        do {
            condition_0 = int32_t(matrix_accumulator_9) >= int32_t(matrix_accumulator_8);
            if (!condition_0) {
                matrix_fragment_10 = matrix_fragment_7 * matrix_fragment_6;

                matrix_fragment_5 = matrix_accumulator_9;
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_12];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_11 = matrix_fragment_10 + matrix_fragment_5;

                    split_address_words(
                        matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

                    matrix_fragment_11 = pipeline_read<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_accumulator_8);

                    condition_0 = int32_t(matrix_fragment_11) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_11 = pipeline_read<CheckBounds>(
                                combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                memory_ranges, error_counts);

                            condition_0 = int32_t(matrix_fragment_11) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            matrix_fragment_7 = matrix_fragment_7 + 0x1u;

            condition_0 = uint32_t(matrix_fragment_7) != uint32_t(matrix_fragment_12);
        } while (condition_0);
    }
    matrix_fragment_6 = parameters[parameter_tensor_height_8];
    matrix_fragment_7 = parameters[parameter_tensor_width_9];

    condition_3 = false;

    __syncwarp(__activemask());

    condition_5 = false;

    matrix_fragment_10 = index_word_148 + 0x1u;

    matrix_fragment_14 = 0u;

    matrix_fragment_12 = 0u;

    matrix_accumulator_17 = 0u;

    matrix_accumulator_2 = matrix_fragment_6 & 0xfffffffcu;

    matrix_accumulator_3 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    condition_0 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + matrix_fragment_6;

    matrix_accumulator_2 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

    index_word_146 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x2u >> 32) + matrix_fragment_7;

    index_word_145 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    if (condition_0) {
        matrix_accumulator_3 = index_word_150 + 0x1u;

        matrix_fragment_5 = index_word_150 + 0x1u;

        condition_4 = int32_t(index_word_150) >= int32_t(index_word_146);

        condition_6 = int32_t(index_word_150) >= int32_t(index_word_146);

        matrix_fragment_14 = index_word_150 * index_word_145 + index_word_145;

        condition_1 = int32_t(matrix_accumulator_3) >= int32_t(index_word_146);

        condition_2 = int32_t(matrix_fragment_5) >= int32_t(index_word_146);

        condition_3 = int32_t(index_word_0) < int32_t(0u - 0x3u) || condition_4;
    }

    condition_4 = false;

    if (condition_0) {
        matrix_fragment_12 = index_word_150 * index_word_145;

        condition_4 = int32_t(index_word_0) < int32_t(0u - 0x7u) || condition_1;
    }

    condition_1 = false;

    if (condition_0) {
        matrix_accumulator_17 = index_word_150 * index_word_145 + index_word_145;

        condition_5 = int32_t(index_word_0) < int32_t(0u - 0x3u) || condition_6;

        condition_1 = int32_t(index_word_0) < int32_t(0u - 0x7u) || condition_2;
    }

    condition_6 = int32_t(index_word_148) >= int32_t(index_word_145);

    index_word_0 = matrix_fragment_7 & 0xfffffffcu;

    condition_2 = true;

    if (condition_0) {
        condition_2 = !condition_3;
    }

    matrix_accumulator_3 = condition_3 ? index_word_148 : 0u;

    condition_6 = int32_t(matrix_fragment_4) > int32_t(0u - 0x4u) && !condition_6;

    condition_3 = uint32_t(index_word_0) == uint32_t(0x4u) || condition_3;

    matrix_fragment_5 = condition_5 ? matrix_fragment_10 : 0u;

    matrix_fragment_7 = condition_4 ? index_word_148 : 0u;

    index_word_151 = condition_6 ? 0x40u : 0u;

    condition_2 = condition_2 & (condition_3 | condition_6);

    condition_3 = int32_t(matrix_fragment_10) >= int32_t(index_word_145);

    matrix_accumulator_2 = uint32_t(condition_2) << 2;

    condition_2 = int32_t(matrix_fragment_4) > int32_t(0u - 0x8u) && !condition_3;

    condition_3 = true;

    if (condition_0) {
        condition_3 = !condition_5;
    }

    condition_5 = uint32_t(index_word_0) == uint32_t(0x4u) || condition_5;

    index_word_149 = uint32_t(condition_2) << 2;

    condition_3 = condition_3 & (condition_5 | condition_2);

    condition_5 = true;

    if (condition_0) {
        condition_5 = !condition_4;
    }

    condition_4 = uint32_t(index_word_0) == uint32_t(0x4u) || condition_4;

    condition_4 = condition_5 & (condition_4 | condition_6);

    condition_5 = true;

    condition_6 = uint32_t(index_word_0) == uint32_t(0x4u) || condition_1;

    if (condition_0) {
        condition_5 = !condition_1;
    }

    if (!condition_3) {
        matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    condition_5 = condition_5 & (condition_6 | condition_2);

    condition_2 = int32_t(matrix_accumulator_2) != int32_t(0u);

    condition_6 = uint32_t(index_word_0) != uint32_t(0x4u);

    index_word_0 = 0u;

    if (condition_0) {
        index_word_0 = index_word_150 * index_word_145;
    }

    if (!condition_4) {
        matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_11 = !condition_6 ? matrix_accumulator_3 : index_word_148;

    matrix_fragment_15 = !condition_6 ? matrix_fragment_7 : index_word_148;

    matrix_fragment_13 = !condition_6 ? matrix_fragment_5 : matrix_fragment_10;

    if (condition_4) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];
    }

    if (!condition_3) {
        matrix_accumulator_19 =
            permute_signed(matrix_accumulator_19, matrix_accumulator_19, 0x5410u);
    }

    if (condition_4) {
        matrix_fragment_15 = matrix_fragment_15 + matrix_fragment_14;
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 + index_word_0;
    }

    if (!condition_6) {
        matrix_fragment_10 = condition_1 ? matrix_fragment_10 : 0u;
    }

    index_word_0 = threadIdx.x & 31u;

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_13 + matrix_fragment_12;
    }

    if (condition_2) {
        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        matrix_fragment_15 = matrix_fragment_15 * 0x8u + matrix_accumulator_136;
    }

    if (condition_5) {
        matrix_accumulator_17 = matrix_fragment_10 + matrix_accumulator_17;
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 * 0x8u + matrix_accumulator_136;
    }

    if (!condition_5) {
        matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_4) {
        matrix_fragment_15 = matrix_fragment_15 * 0x80u;
    }

    if (!condition_4) {
        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_23, matrix_accumulator_23, 0x5410u);
    }

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_13 * 0x8u + matrix_accumulator_136;

        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        matrix_accumulator_17 = matrix_accumulator_17 * 0x8u + matrix_accumulator_136;
    }

    if (!condition_5) {
        matrix_fragment_35 = permute_signed(matrix_fragment_35, matrix_fragment_35, 0x5410u);
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 * 0x80u;
    }

    if (!condition_3) {
        matrix_accumulator_16 = matrix_accumulator_19;
    }

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_13 * 0x80u;
    }

    if (!condition_3) {
        matrix_accumulator_18 = matrix_accumulator_19;
    }

    if (condition_5) {
        matrix_accumulator_17 = matrix_accumulator_17 * 0x80u;

        matrix_accumulator_8 = parameters[parameter_input_low_0];
        matrix_accumulator_9 = parameters[parameter_input_high_1];
    }

    if (condition_4) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_2) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_4) {
        matrix_accumulator_20 = matrix_accumulator_23;

        matrix_accumulator_21 = matrix_accumulator_23;
    }

    if (!condition_2) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (!condition_4) {
        matrix_accumulator_22 = matrix_accumulator_23;
    }

    if (!condition_5) {
        matrix_fragment_32 = matrix_fragment_35;
    }

    if (!condition_2) {
        matrix_fragment_12 = matrix_fragment_15;

        matrix_fragment_14 = matrix_fragment_15;
    }

    if (!condition_5) {
        matrix_fragment_33 = matrix_fragment_35;

        matrix_fragment_34 = matrix_fragment_35;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (!condition_2) {
        matrix_fragment_13 = matrix_fragment_15;
    }

    __syncthreads();

    if (condition_4) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (condition_5) {
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));
    }

    if (!condition_3) {
        matrix_accumulator_17 = matrix_accumulator_19;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_5) {
        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_8, matrix_accumulator_9));
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_accumulator_16,
            matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19, memory_ranges,
            error_counts);
    }

    if (condition_4) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7), matrix_accumulator_20,
            matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
            error_counts);
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_8, matrix_accumulator_9), matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);
    }

    uniform_index_word_5 = 0u;

    uniform_index_word_4 = 0x400u;

    uniform_parameter_word_6 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_7 = parameters[parameter_trained_weights_high_5];

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    packed_half_30 = 0;
    matrix_accumulator_31 = 0;

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_accumulator_26 = 0;
    matrix_accumulator_27 = 0;

    matrix_accumulator_24 = 0;
    matrix_accumulator_25 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_accumulator_8 = 0;
    matrix_accumulator_9 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    split_address_words(matrix_accumulator_140, matrix_accumulator_141,
                        combine_words(uniform_parameter_word_6, uniform_parameter_word_7));

    uniform_index_word_5 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

    uniform_index_word_4 = 0u;

    matrix_fragment_11 = (matrix_accumulator_136 << 0x9u) + uniform_index_word_5;

    tile_word_144 = index_word_0 * 0x10u + matrix_fragment_11;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x1000u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x2000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x3000u, matrix_fragment_32,
                                         matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_accumulator_18 = 0;
    matrix_accumulator_19 = 0;

    matrix_accumulator_16 = 0u;

    matrix_accumulator_17 = 0u;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    __syncthreads();
    do {
        split_address_words(address_word_142, address_word_143,
                            uint64_t(uint32_t(matrix_accumulator_136)) *
                                    uint64_t(uint32_t(0x8000u)) +
                                combine_words(matrix_accumulator_140, matrix_accumulator_141));

        split_address_words(address_word_142, address_word_143,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_142, address_word_143));

        load_four_words_cached<CheckBounds>(combine_words(address_word_142, address_word_143),
                                            matrix_fragment_88, matrix_fragment_89,
                                            matrix_fragment_90, matrix_fragment_91, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x200ull, matrix_fragment_92,
            matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x1000ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x1200ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x2000ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x2200ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x3000ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x3200ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        matrix_accumulator_137 = (index_word_0 << 0x4u) + uniform_index_word_5;

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x4000ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_137,
                                            matrix_accumulator_112, matrix_accumulator_113,
                                            matrix_accumulator_114, matrix_accumulator_115,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x4200ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1000u, matrix_fragment_68, matrix_fragment_69,
            matrix_fragment_70, matrix_fragment_71, shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x5000ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2000u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x200u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1200u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3200u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_90,
                                 matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_94,
                                 matrix_fragment_95, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x5200ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2200u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x6000ull, matrix_fragment_88,
            matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x6200ull, matrix_fragment_92,
            matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x400u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_102,
                                 matrix_accumulator_103);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_137 + 0x1400u,
                                            matrix_accumulator_100, matrix_accumulator_101,
                                            matrix_accumulator_102, matrix_accumulator_103,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x7000ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x600u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_120,
                                 matrix_accumulator_121);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3400u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2400u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_32,
                                 matrix_fragment_33);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_142, address_word_143) + 0x7200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(
            matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_40, matrix_fragment_41, matrix_accumulator_130, matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_42,
                                 matrix_fragment_43, matrix_fragment_72, matrix_fragment_73);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3600u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_74,
                                 matrix_fragment_75);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1600u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2600u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_110, matrix_accumulator_111, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_44, matrix_fragment_45, matrix_accumulator_110, matrix_accumulator_111);

        multiply_accumulate_half(
            matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_46, matrix_fragment_47, matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x800u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3800u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2800u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0xa00u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1800u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3a00u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1a00u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2a00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_126,
                                 matrix_accumulator_127);

        split_address_words(matrix_fragment_44, matrix_fragment_45,
                            uint64_t(uint32_t(matrix_accumulator_136)) *
                                    uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_accumulator_140, matrix_accumulator_141));

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_116,
                                 matrix_accumulator_117);

        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_44, matrix_fragment_45));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x40000ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x40200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_112,
                                 matrix_accumulator_113);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2c00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1e00u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x1c00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0xc00u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_90, matrix_fragment_91, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0xe00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_90, matrix_fragment_91, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3c00u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_90, matrix_fragment_91, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x2e00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_90, matrix_fragment_91, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_128,
                                 matrix_accumulator_129);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_132), 0x44004400u);

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_133), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_130,
                                 matrix_accumulator_131);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_65 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_63 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_62 = 0x2b28u;

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_134,
                                 matrix_accumulator_135);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_63),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_137 + 0x3e00u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_65),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_64 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_fragment_63 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_106,
                                 matrix_accumulator_107);

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_65 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_112,
                                 matrix_accumulator_113);

        matrix_fragment_65 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0xc400c400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_63),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_66 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_65),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_130), 0x44004400u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_138), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_131), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65),
            select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_139), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_110), 0x44004400u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_111), 0x44004400u);

        matrix_fragment_37 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_37),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_114,
                                 matrix_accumulator_115);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

        matrix_accumulator_139 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_124,
                                 matrix_accumulator_125);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_32 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_118), 0x44004400u);

        matrix_fragment_33 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_94,
                                 matrix_fragment_95);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

        matrix_fragment_37 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_33),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                select_half_operand<2, false, false>(matrix_fragment_32));

        matrix_fragment_32 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_108), 0x44004400u);

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_34 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x44004400u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_119), 0x44004400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_33 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_109), 0x44004400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_32 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

        matrix_fragment_35 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_115), 0x44004400u);

        matrix_fragment_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0xc400c400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_34 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

        matrix_fragment_33 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_32),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32),
            select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

        matrix_fragment_32 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_35 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_34),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_accumulator_127 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_65 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_33 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_138 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34),
            select_half_operand<2, false, false>(matrix_fragment_35), 0x3b283b28u);

        matrix_fragment_34 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_59 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_35 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_32 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(matrix_fragment_33));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_33 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(matrix_fragment_34));

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

        matrix_fragment_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

        matrix_fragment_34 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(matrix_fragment_35));

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_35 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

        matrix_fragment_36 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_48 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_124), 0x44004400u);

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_125), 0x44004400u);

        matrix_fragment_64 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_66 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

        matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_64),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_66),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_49 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(matrix_fragment_62), 0x37283728u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_fragment_65 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

        matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

        matrix_fragment_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                          select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_56 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_57 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_58 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                          select_half_operand<2, false, false>(matrix_fragment_60));

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_104,
                                 matrix_accumulator_105);

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0x60u);

        uniform_parameter_word_6 = uniform_index_word_4 + 0x20u;

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_46, matrix_fragment_47, packed_half_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_32,
                                 matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_42, matrix_fragment_43, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_2,
                                 matrix_accumulator_3);

        split_address_words(matrix_accumulator_140, matrix_accumulator_141,
                            combine_words(matrix_accumulator_140, matrix_accumulator_141) + 0x400u);

        uniform_index_word_4 = uniform_parameter_word_6;
    } while (!uniform_condition_0);
    address_word_160 = parameters[parameter_trained_weights_low_4];
    address_word_161 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_38 = matrix_accumulator_136 * 0x20u;

    index_word_156 = index_word_0;

    index_word_155 = uint32_t(uint32_t(matrix_fragment_38) >> 0x1u);

    matrix_fragment_37 = index_word_156 & 0x7ffffffcu;

    index_word_157 = index_word_156 & 0xfffffffcu;

    matrix_fragment_34 = index_word_155 + 0x8u;

    matrix_fragment_39 = 0u - matrix_fragment_37 + index_word_0;

    matrix_fragment_36 = index_word_155 + 0xcu;

    packed_half_152 = index_word_155 + 0x4u;

    matrix_fragment_35 = matrix_fragment_34 + index_word_0 + (0u - index_word_157);

    matrix_fragment_39 = matrix_fragment_39 * 0x2u + matrix_fragment_38;

    matrix_fragment_41 = matrix_fragment_36 + index_word_0 + (0u - index_word_157);

    packed_half_152 = packed_half_152 + index_word_0 + (0u - index_word_157);

    split_address_words(matrix_fragment_34, matrix_fragment_35,
                        uint64_t(uint32_t(matrix_fragment_35)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_39 = uint32_t(int32_t(matrix_fragment_39) >> 0x1u);

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        uint64_t(uint32_t(matrix_fragment_41)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_47 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_34, matrix_fragment_35) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_32, matrix_fragment_33,
                        uint64_t(uint32_t(packed_half_152)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_46 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_36, matrix_fragment_37) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(packed_half_162, address_word_163,
                        int64_t(int32_t(matrix_fragment_39)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_45 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_32, matrix_fragment_33) + 0x58010ull, memory_ranges,
        error_counts);

    matrix_fragment_44 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_162, address_word_163) + 0x58010ull, memory_ranges, error_counts);

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                       select_half_operand<2, false, false>(0u));

    packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_144, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_104 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                      select_half_operand<2, false, false>(matrix_fragment_39));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x1000u, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                                       select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(matrix_accumulator_19));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x2000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x3000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                                       select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(matrix_fragment_10));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x1000u, matrix_fragment_40,
                                         matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x2000u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x3000u, matrix_fragment_48,
                                         matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144, matrix_accumulator_104, matrix_accumulator_105,
        matrix_accumulator_106, matrix_accumulator_107, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_8 = unpack_fp8_bits(matrix_fragment_34);

    matrix_accumulator_9 = unpack_fp8_bits(matrix_fragment_35);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_32);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_fragment_38);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_accumulator_26 = unpack_fp8_bits(matrix_accumulator_20);

    matrix_fragment_6 = unpack_fp8_bits(matrix_accumulator_20 >> 16);

    matrix_accumulator_27 = unpack_fp8_bits(matrix_accumulator_21);

    matrix_fragment_7 = unpack_fp8_bits(matrix_accumulator_21 >> 16);

    matrix_accumulator_18 = unpack_fp8_bits(matrix_accumulator_22);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_fragment_39);

    matrix_accumulator_20 = unpack_fp8_bits(matrix_fragment_14);

    matrix_accumulator_21 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_accumulator_19 = unpack_fp8_bits(matrix_accumulator_23);

    matrix_accumulator_28 = unpack_fp8_bits(matrix_fragment_12);

    matrix_accumulator_29 = unpack_fp8_bits(matrix_fragment_13);

    packed_half_30 = unpack_fp8_bits(matrix_fragment_15);

    matrix_accumulator_31 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    matrix_accumulator_24 = unpack_fp8_bits(matrix_fragment_36);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_36 >> 16);

    matrix_accumulator_25 = unpack_fp8_bits(matrix_fragment_37);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_37 >> 16);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_38 >> 16);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_39 >> 16);

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_22 >> 16);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_23 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    uniform_index_word_4 = 0u;

    uniform_parameter_word_6 = uniform_index_word_5;

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_8),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_9),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                        uint64_t(uint32_t(matrix_accumulator_136)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        combine_words(matrix_accumulator_8, matrix_accumulator_9) + 0x48200u);

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_46));

    packed_half_30 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                         select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    __syncthreads();
    do {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_22, matrix_accumulator_23) - 0x200ull,
            matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_10, matrix_fragment_11,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_22, matrix_accumulator_23), matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        matrix_fragment_42 = (index_word_0 << 0x4u) + uniform_parameter_word_6;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_42, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_6 = uniform_parameter_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_42 + 0x1000u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_42 + 0x2000u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_42 + 0x3000u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_8, matrix_accumulator_9, packed_half_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_accumulator_8, matrix_accumulator_9, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                                 matrix_fragment_41);
    } while (!uniform_condition_0);
    matrix_accumulator_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                         select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                        uint64_t(uint32_t(matrix_accumulator_136)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                      select_half_operand<2, false, false>(matrix_accumulator_9));

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                        combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x58c20u);

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = 0;
    matrix_fragment_35 = 0;

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                      select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_30 = 0;
    matrix_accumulator_31 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_32 = 0;
    matrix_fragment_33 = 0;

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_24 = 0;
    matrix_accumulator_25 = 0;

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144, matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_10,
        matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x1000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_26 = 0;
    matrix_accumulator_27 = 0;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x2000u, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_28 = 0;
    matrix_accumulator_29 = 0;

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_accumulator_8 = 0;
    matrix_accumulator_9 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x3000u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_54 = 0;
    matrix_fragment_55 = 0;

    matrix_fragment_56 = 0;
    matrix_fragment_57 = 0;

    matrix_fragment_58 = 0;
    matrix_fragment_59 = 0;

    matrix_accumulator_104 = 0;
    matrix_accumulator_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_accumulator_128 = 0;
    matrix_accumulator_129 = 0;

    matrix_accumulator_130 = 0;
    matrix_accumulator_131 = 0;

    matrix_accumulator_132 = 0;
    matrix_accumulator_133 = 0;

    matrix_accumulator_134 = 0;
    matrix_accumulator_135 = 0;

    matrix_accumulator_136 = 0;
    matrix_accumulator_137 = 0;

    matrix_accumulator_138 = 0;
    matrix_accumulator_139 = 0;

    matrix_accumulator_18 = 0;
    matrix_accumulator_19 = 0;

    matrix_accumulator_16 = 0;
    matrix_accumulator_17 = 0;

    matrix_accumulator_140 = 0;
    matrix_accumulator_141 = 0;

    uniform_index_word_4 = 0u;

    uniform_parameter_word_6 = uniform_index_word_5;

    __syncthreads();
    do {
        split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) - 0xa00ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) - 0x800ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) - 0x600ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) - 0x400ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) - 0x200ull,
            matrix_fragment_76, matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103), matrix_fragment_80,
            matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges,
            error_counts);

        address_word_142 = (index_word_0 << 0x4u) + uniform_parameter_word_6;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            combine_words(matrix_accumulator_100, matrix_accumulator_101) +
                                0x6000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, address_word_142, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        uniform_parameter_word_6 = uniform_parameter_word_6 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, address_word_142 + 0x1000u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, address_word_142 + 0x2000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, address_word_142 + 0x3000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_54,
                                 matrix_fragment_55);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_28,
                                 matrix_accumulator_29);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, packed_half_30,
                                 matrix_accumulator_31);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_48,
                                 matrix_fragment_49);
    } while (!uniform_condition_0);
    thread_row_158 = threadIdx.y;

    split_address_words(matrix_fragment_74, matrix_fragment_75,
                        uint64_t(uint32_t(thread_row_158)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_74 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_74, matrix_fragment_75) + 0x98220ull, memory_ranges,
        error_counts);

    address_word_165 = thread_row_158 * 0x800u;

    __syncthreads();

    split_address_words(address_word_164, address_word_165,
                        uint64_t(uint32_t(address_word_165)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    split_address_words(address_word_164, address_word_165,
                        int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_164, address_word_165));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88220ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88420ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    matrix_fragment_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88820ull, matrix_fragment_96,
        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    {
        const uint32_t bits = index_word_0;
        condition_0 = bool(bits & 1u);
        condition_1 = bool(bits & 2u);
        condition_3 = bool(bits & 8u);
        condition_4 = bool(bits & 16u);
    }

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_3 = transpose_half_tile(matrix_accumulator_3);

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_61 = result_word;
    }

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_5 = transpose_half_tile(matrix_fragment_5);

    address_word_143 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    address_word_142 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_4 = transpose_half_tile(matrix_fragment_4);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_7 = transpose_half_tile(matrix_fragment_7);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_107 = transpose_half_tile(matrix_accumulator_107);

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_105 = transpose_half_tile(matrix_accumulator_105);

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_59 = transpose_half_tile(matrix_fragment_59);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_106 = transpose_half_tile(matrix_accumulator_106);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_104 = transpose_half_tile(matrix_accumulator_104);

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                      select_half_operand<2, false, false>(matrix_accumulator_118),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_58 = transpose_half_tile(matrix_fragment_58);

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                      select_half_operand<2, false, false>(matrix_accumulator_119),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    packed_half_162 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_162, address_word_163) + 0xa8240ull, memory_ranges, error_counts);

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88620ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_90 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_142), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_143), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_89 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_81 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    address_word_142 = add_half_pairs(select_half_operand<2, false, false>(address_word_142),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    address_word_143 = add_half_pairs(select_half_operand<2, false, false>(address_word_143),
                                      select_half_operand<2, false, false>(matrix_fragment_88));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_84 = permute_signed(matrix_fragment_81, matrix_fragment_81, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_143), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_85 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5432u);

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(address_word_142), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    address_word_143 = add_half_pairs(select_half_operand<2, false, false>(address_word_143),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_82 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_70 = permute_signed(matrix_fragment_71, matrix_fragment_71, 0x5432u);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_70 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_80 = permute_signed(matrix_fragment_75, matrix_fragment_75, 0x5432u);

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_69), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_80 = result_word;
    }

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x4100410u);

    address_word_142 = add_half_pairs(select_half_operand<2, false, false>(address_word_142),
                                      select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88a20ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_81 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_81 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_76 = result_word;
    }

    matrix_fragment_77 = half_to_float_bits((matrix_fragment_88 & 65535u) * 65537u);

    matrix_fragment_78 = half_to_float_bits((matrix_fragment_88 >> 16) * 65537u);

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_71 >> 16) * 65537u);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_71 & 65535u) * 65537u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x4100410u);

    matrix_fragment_71 = rsqrt_bits(matrix_fragment_90);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                     select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_69 = permute_signed(matrix_fragment_70, matrix_fragment_70, 0x5432u);

    matrix_fragment_88 = rsqrt_bits(matrix_fragment_88);

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x4100410u);

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_89 = half_to_float_bits((matrix_fragment_75 & 65535u) * 65537u);

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x4100410u);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_75 >> 16) * 65537u);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_91 = half_to_float_bits((matrix_fragment_81 & 65535u) * 65537u);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                     select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_81 = half_to_float_bits((matrix_fragment_81 >> 16) * 65537u);

    matrix_fragment_79 = half_to_float_bits((matrix_fragment_82 & 65535u) * 65537u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x4100410u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_82 = half_to_float_bits((matrix_fragment_82 >> 16) * 65537u);

    matrix_fragment_76 = rsqrt_bits(matrix_fragment_91);

    matrix_fragment_92 = pack_half(matrix_fragment_88, matrix_fragment_71);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88c20ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_fragment_81 = rsqrt_bits(matrix_fragment_81);

    matrix_fragment_80 = rsqrt_bits(matrix_fragment_89);

    matrix_fragment_91 = rsqrt_bits(matrix_fragment_90);

    matrix_fragment_89 = half_to_float_bits((matrix_fragment_83 & 65535u) * 65537u);

    matrix_fragment_79 = rsqrt_bits(matrix_fragment_79);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_83 >> 16) * 65537u);

    matrix_fragment_88 = rsqrt_bits(matrix_fragment_82);

    matrix_fragment_77 = rsqrt_bits(matrix_fragment_77);

    matrix_fragment_78 = rsqrt_bits(matrix_fragment_78);

    matrix_fragment_89 = rsqrt_bits(matrix_fragment_89);

    matrix_fragment_90 = rsqrt_bits(matrix_fragment_90);

    matrix_fragment_93 = pack_half(matrix_fragment_81, matrix_fragment_76);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_91 = pack_half(matrix_fragment_91, matrix_fragment_80);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_88 = pack_half(matrix_fragment_88, matrix_fragment_79);

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x88e20ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = pack_half(matrix_fragment_78, matrix_fragment_77);

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_79 = pack_half(matrix_fragment_90, matrix_fragment_89);

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_88));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89020ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_accumulator_118 = pack_half(0u, matrix_fragment_74);

    matrix_fragment_74 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                       select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_73 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5432u);

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x4100410u);

    matrix_accumulator_122 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                      select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_133 = half_to_float_bits((matrix_accumulator_130 & 65535u) * 65537u);

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 = half_to_float_bits((matrix_accumulator_130 >> 16) * 65537u);

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_128 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_133 = rsqrt_bits(matrix_accumulator_133);

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x4100410u);

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                                       select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_accumulator_122 = rsqrt_bits(matrix_accumulator_134);

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_136 = half_to_float_bits((matrix_accumulator_132 >> 16) * 65537u);

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_136);

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_124 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                      select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89220ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    matrix_fragment_65 = half_to_float_bits((matrix_accumulator_132 & 65535u) * 65537u);

    matrix_accumulator_133 = pack_half(matrix_accumulator_122, matrix_accumulator_133);

    matrix_fragment_65 = rsqrt_bits(matrix_fragment_65);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = permute_signed(packed_half_153, packed_half_153, 0x5432u);

    matrix_accumulator_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                      select_half_operand<2, false, false>(matrix_fragment_79));

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_accumulator_121 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                      select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_125 = permute_signed(packed_half_154, packed_half_154, 0x5432u);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x4100410u);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                     select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_110 = pack_half(matrix_accumulator_134, matrix_fragment_65);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_134 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(matrix_accumulator_132));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    packed_half_154 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89420ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_accumulator_100 = half_to_float_bits((matrix_accumulator_112 & 65535u) * 65537u);

    matrix_accumulator_101 = half_to_float_bits((matrix_accumulator_112 >> 16) * 65537u);

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 = rsqrt_bits(matrix_accumulator_100);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_accumulator_132 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_101 = rsqrt_bits(matrix_accumulator_101);

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_102 = half_to_float_bits((packed_half_154 & 65535u) * 65537u);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    packed_half_154 = half_to_float_bits((packed_half_154 >> 16) * 65537u);

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89620ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_accumulator_116 = rsqrt_bits(matrix_accumulator_102);

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_117 = rsqrt_bits(packed_half_154);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_accumulator_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                      select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_61 = pack_half(matrix_accumulator_101, matrix_accumulator_100);

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_102 = permute_signed(address_word_143, address_word_143, 0x5432u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_62 = pack_half(matrix_accumulator_117, matrix_accumulator_116);

    matrix_fragment_63 = permute_signed(address_word_142, address_word_142, 0x5432u);

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    address_word_143 = add_half_pairs(select_half_operand<2, false, false>(address_word_143),
                                      select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_116 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                      select_half_operand<2, false, false>(matrix_fragment_61));

    address_word_142 = add_half_pairs(select_half_operand<2, false, false>(address_word_142),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_accumulator_117 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                           select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    address_word_143 =
        maximum_half_pairs(select_half_operand<2, false, false>(address_word_143), 0x4100410u);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89820ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    address_word_142 =
        maximum_half_pairs(select_half_operand<2, false, false>(address_word_142), 0x4100410u);

    matrix_fragment_60 = half_to_float_bits((address_word_143 & 65535u) * 65537u);

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                      select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_61 = half_to_float_bits((address_word_143 >> 16) * 65537u);

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_40 = half_to_float_bits((address_word_142 & 65535u) * 65537u);

    matrix_fragment_60 = rsqrt_bits(matrix_fragment_60);

    address_word_142 = half_to_float_bits((address_word_142 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89a20ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    matrix_fragment_61 = rsqrt_bits(matrix_fragment_61);

    matrix_fragment_40 = rsqrt_bits(matrix_fragment_40);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_95 = rsqrt_bits(address_word_142);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_fragment_94 = pack_half(matrix_fragment_61, matrix_fragment_60);

    matrix_fragment_41 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_40 = pack_half(matrix_fragment_95, matrix_fragment_40);

    matrix_fragment_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_93 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89c20ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_41), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_57), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_94 = result_word;
    }

    matrix_fragment_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_50 = permute_signed(matrix_fragment_51, matrix_fragment_51, 0x5432u);

    matrix_fragment_57 = permute_signed(matrix_fragment_94, matrix_fragment_94, 0x5432u);

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                        select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_94 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x4100410u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x4100410u);

    matrix_fragment_93 = half_to_float_bits((matrix_fragment_56 & 65535u) * 65537u);

    matrix_fragment_95 = half_to_float_bits((matrix_fragment_56 >> 16) * 65537u);

    matrix_fragment_57 = half_to_float_bits((matrix_fragment_94 & 65535u) * 65537u);

    matrix_fragment_94 = half_to_float_bits((matrix_fragment_94 >> 16) * 65537u);

    matrix_fragment_93 = rsqrt_bits(matrix_fragment_93);

    matrix_fragment_92 = rsqrt_bits(matrix_fragment_95);

    matrix_fragment_57 = rsqrt_bits(matrix_fragment_57);

    matrix_fragment_94 = rsqrt_bits(matrix_fragment_94);

    matrix_fragment_93 = pack_half(matrix_fragment_92, matrix_fragment_93);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_94 = pack_half(matrix_fragment_94, matrix_fragment_57);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_70,
                             matrix_fragment_71);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x89e20ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_80,
                             matrix_fragment_81);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_164, address_word_165) + 0x8a020ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    matrix_fragment_56 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                                             select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_119 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_92 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                       select_half_operand<2, false, false>(matrix_accumulator_119));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_56 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_27),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_95 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_56), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_57), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_accumulator_16));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_16 = result_word;
    }

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_17 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5432u);

    matrix_fragment_10 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x4100410u);

    matrix_accumulator_16 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                       select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_accumulator_17 = half_to_float_bits((matrix_fragment_10 & 65535u) * 65537u);

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_10 >> 16) * 65537u);

    matrix_fragment_11 = permute_signed(matrix_accumulator_16, matrix_accumulator_16, 0x5432u);

    matrix_accumulator_17 = rsqrt_bits(matrix_accumulator_17);

    matrix_accumulator_16 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    matrix_accumulator_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16), 0x4100410u);

    matrix_fragment_11 = half_to_float_bits((matrix_accumulator_16 & 65535u) * 65537u);

    matrix_accumulator_16 = half_to_float_bits((matrix_accumulator_16 >> 16) * 65537u);

    matrix_fragment_11 = rsqrt_bits(matrix_fragment_11);

    matrix_accumulator_17 = pack_half(matrix_fragment_10, matrix_accumulator_17);

    matrix_accumulator_16 = rsqrt_bits(matrix_accumulator_16);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    packed_half_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_72,
                             matrix_fragment_73);

    packed_half_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_56 = pack_half(matrix_accumulator_16, matrix_fragment_11);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_64,
                             matrix_fragment_65);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    packed_half_30 = 0x29c0u;

    matrix_accumulator_130 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_136 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x3c203c20u);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_122 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3c203c20u);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_124 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3c203c20u);

    matrix_accumulator_131 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_136 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136), 0x3c203c20u);

    matrix_accumulator_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_137 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_130 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x3e473e47u);

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_122 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_122), 0x3e473e47u);

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_124 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3e473e47u);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_136 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_136), 0x3e473e47u);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_accumulator_131 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x3c203c20u);

    matrix_accumulator_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_123 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3c203c20u);

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_125 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3c203c20u);

    matrix_accumulator_137 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137), 0x3c203c20u);

    matrix_accumulator_110 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3c203c20u);

    matrix_accumulator_114 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3c203c20u);

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<0, false, false>(matrix_accumulator_118));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_131 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x3e473e47u);

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_123 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_123), 0x3e473e47u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_125 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3e473e47u);

    matrix_accumulator_137 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_137), 0x3e473e47u);

    matrix_fragment_94 = (matrix_accumulator_130 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 = (matrix_accumulator_122 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 = (matrix_accumulator_124 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_31 = (matrix_accumulator_136 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3e473e47u);

    matrix_accumulator_114 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3e473e47u);

    matrix_accumulator_111 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3c203c20u);

    matrix_accumulator_115 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3c203c20u);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_32 = (matrix_accumulator_131 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = (matrix_accumulator_123 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (matrix_accumulator_125 << 0x5u) + 0x7ff88000u;

    matrix_fragment_92 = (matrix_accumulator_137 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_16 = (matrix_accumulator_110 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_27 = (matrix_accumulator_114 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_111 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3e473e47u);

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    matrix_accumulator_115 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3e473e47u);

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    matrix_accumulator_26 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                        select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    matrix_accumulator_29 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                       select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_17 = (matrix_accumulator_111 << 0x5u) + 0x7ff88000u;

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_28 = (matrix_accumulator_115 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_118 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_10 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                       select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_11 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                       select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_26 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_29 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3c203c20u);

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                       select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                       select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3e473e47u);

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_95 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_97 = (matrix_fragment_85 << 0x5u) + 0x7ff88000u;

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_98 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_96 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_85 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_fragment_84 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = (matrix_fragment_51 << 0x5u) + 0x7ff88000u;

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3c203c20u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3c203c20u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_81 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3e473e47u);

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_83 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_fragment_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3e473e47u);

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_51 = (matrix_fragment_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 = (matrix_fragment_83 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_81 = (matrix_fragment_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_41 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_82 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_83 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                       select_half_operand<2, false, false>(matrix_fragment_90));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_50 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_80 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_accumulator_118 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_76 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_73 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_79 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_75 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_74 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_77 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_88 = condition_0 ? matrix_accumulator_110 : matrix_accumulator_115;

    matrix_fragment_89 = condition_0 ? matrix_accumulator_111 : matrix_accumulator_118;

    matrix_fragment_64 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_115 = condition_0 ? matrix_accumulator_115 : matrix_accumulator_110;

    matrix_fragment_66 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                       select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_65 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_78 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_102 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3c203c20u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3c203c20u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_72 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_116 = !condition_1 ? matrix_fragment_88 : matrix_fragment_89;

    matrix_accumulator_117 = !condition_1 ? matrix_fragment_89 : matrix_fragment_88;

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_102 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3e473e47u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_accumulator_103 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3e473e47u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_67 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3c203c20u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_118 = condition_0 ? matrix_accumulator_118 : matrix_accumulator_111;

    matrix_fragment_63 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_61 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    packed_half_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<0, false, false>(packed_half_30), 0x3d343d34u);

    matrix_fragment_60 = (matrix_accumulator_102 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = (matrix_accumulator_103 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3e473e47u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_accumulator_111 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_101 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_102 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_90 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_accumulator_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3c203c20u);

    packed_half_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x3c203c20u);

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3e473e47u);

    matrix_fragment_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                       select_half_operand<2, false, false>(matrix_fragment_91));

    packed_half_30 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_30), 0x3e473e47u);

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                       select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_111 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_accumulator_110 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_113 = (matrix_accumulator_113 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_112 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_103 = (matrix_fragment_53 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_71 = (matrix_fragment_54 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = index_word_0 * 0x4u;

    matrix_accumulator_114 = (packed_half_30 << 0x5u) + 0x7ff88000u;

    matrix_fragment_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                       select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_15 = matrix_fragment_14 & 0x1cu;

    packed_half_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                    select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_36 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_15 =
        uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1du) >> 32) + matrix_fragment_15;

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_14 = !condition_1 ? matrix_accumulator_115 : matrix_accumulator_118;

    matrix_fragment_36 = condition_0 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_fragment_37 = condition_0 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_accumulator_115 = !condition_1 ? matrix_accumulator_118 : matrix_accumulator_115;

    matrix_fragment_13 = 0x1u ^ matrix_fragment_15;

    matrix_fragment_12 = 0x2u ^ matrix_fragment_15;

    packed_half_30 = 0x3u ^ matrix_fragment_15;

    matrix_fragment_39 = condition_0 ? matrix_fragment_35 : matrix_fragment_34;

    matrix_fragment_38 = condition_0 ? matrix_fragment_34 : matrix_fragment_35;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_116 = result_word;
    }

    matrix_fragment_52 = !condition_1 ? matrix_fragment_36 : matrix_fragment_39;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    matrix_fragment_39 = !condition_1 ? matrix_fragment_39 : matrix_fragment_36;

    matrix_fragment_54 = !condition_1 ? matrix_fragment_37 : matrix_fragment_38;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_117), "r"(matrix_fragment_12), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    matrix_fragment_53 = !condition_1 ? matrix_fragment_38 : matrix_fragment_37;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(packed_half_30), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_52), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_52 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_54), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_55 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_39), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_53), "r"(packed_half_30), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    matrix_fragment_14 = !condition_3 ? matrix_accumulator_116 : matrix_fragment_35;

    matrix_fragment_35 = !condition_3 ? matrix_fragment_35 : matrix_accumulator_116;

    matrix_fragment_37 = !condition_3 ? matrix_accumulator_117 : matrix_fragment_34;

    matrix_fragment_34 = !condition_3 ? matrix_fragment_34 : matrix_accumulator_117;

    matrix_fragment_54 = uint32_t(int32_t(index_word_156) >> 0x2u);

    matrix_fragment_36 = !condition_4 ? matrix_fragment_14 : matrix_fragment_37;

    matrix_fragment_14 = !condition_4 ? matrix_fragment_37 : matrix_fragment_14;

    matrix_fragment_13 = !condition_4 ? matrix_fragment_35 : matrix_fragment_34;

    matrix_fragment_37 = !condition_3 ? matrix_fragment_52 : matrix_fragment_55;

    matrix_fragment_52 = !condition_3 ? matrix_fragment_55 : matrix_fragment_52;

    matrix_fragment_34 = !condition_4 ? matrix_fragment_34 : matrix_fragment_35;

    matrix_fragment_12 = !condition_3 ? matrix_fragment_39 : matrix_fragment_38;

    matrix_fragment_39 = !condition_3 ? matrix_fragment_38 : matrix_fragment_39;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_36 = matrix_fragment_54 + 0x8u;

    matrix_fragment_35 = uint32_t(int32_t(index_word_156) >> 0x1fu);

    matrix_fragment_15 = !condition_4 ? matrix_fragment_37 : matrix_fragment_12;

    packed_half_30 = !condition_4 ? matrix_fragment_52 : matrix_fragment_39;

    matrix_fragment_12 = !condition_4 ? matrix_fragment_12 : matrix_fragment_37;

    matrix_fragment_53 = uint32_t(int32_t(matrix_fragment_36) >> 0x1fu);

    matrix_fragment_35 =
        uint32_t(combine_words(matrix_fragment_35, 0u) << 0x6u >> 32) + matrix_fragment_54;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(packed_half_30));

    matrix_fragment_39 = !condition_4 ? matrix_fragment_39 : matrix_fragment_52;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_53 =
        uint32_t(combine_words(matrix_fragment_53, 0u) << 0x6u >> 32) + matrix_fragment_36;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_35 = matrix_fragment_35 & 0xffc0u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_53 = matrix_fragment_53 & 0xffc0u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_35 = matrix_fragment_54 + (0u - matrix_fragment_35);

    matrix_fragment_13 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_13),
                                        select_half_operand<1, false, false>(matrix_fragment_13));

    matrix_fragment_53 = matrix_fragment_36 + (0u - matrix_fragment_53);

    matrix_fragment_12 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_15),
                                        select_half_operand<1, false, false>(matrix_fragment_15));

    matrix_fragment_14 = permute_signed(matrix_fragment_35, 0u, 0x8880u);

    matrix_fragment_15 = permute_signed(matrix_fragment_53, 0u, 0x8880u);

    matrix_fragment_12 = permute_signed(matrix_fragment_13, matrix_fragment_12, 0x5410u);

    matrix_fragment_13 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_fragment_15 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_13) >> 0xau);

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    matrix_fragment_15 = uint32_t(uint32_t(matrix_fragment_15) >> 0xau);

    matrix_fragment_14 = matrix_fragment_14 & 0x1fu;

    matrix_fragment_13 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    packed_half_30 = matrix_fragment_15 & 0x1fu;

    matrix_fragment_39 = uint32_t(int64_t(int32_t(index_word_156)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_fragment_15 = matrix_fragment_35 + matrix_fragment_14;

    matrix_fragment_37 = uint32_t(int64_t(int32_t(index_word_156)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_52 = uint32_t(int32_t(matrix_fragment_39) >> 0x1fu);

    matrix_accumulator_115 = reciprocal_bits(matrix_fragment_13);

    matrix_fragment_14 = matrix_fragment_53 + packed_half_30;

    matrix_fragment_34 = matrix_fragment_15 & 0xe0u;

    matrix_fragment_38 = uint32_t(int32_t(matrix_fragment_37) >> 0x1fu);

    matrix_fragment_52 =
        uint32_t(combine_words(matrix_fragment_52, 0u) << 0x6u >> 32) + matrix_fragment_39;

    packed_half_30 = reciprocal_bits(matrix_fragment_12);

    matrix_fragment_55 = uint32_t(int64_t(int32_t(index_word_156)) * (1ll << 0x1eu) >> 32) + 0x20u;

    matrix_fragment_36 = matrix_fragment_14 & 0xe0u;

    matrix_fragment_38 =
        uint32_t(combine_words(matrix_fragment_38, 0u) << 0x6u >> 32) + matrix_fragment_37;

    matrix_fragment_13 = 0u - matrix_fragment_34;

    matrix_fragment_52 = matrix_fragment_52 & 0xffc0u;

    matrix_fragment_34 = uint32_t(int32_t(matrix_fragment_55) >> 0x1fu);

    matrix_fragment_54 = 0u - matrix_fragment_36;

    matrix_fragment_38 = matrix_fragment_38 & 0xffc0u;

    matrix_fragment_36 = permute_signed(matrix_fragment_13, 0u, 0x7710u);

    matrix_fragment_13 = matrix_fragment_39 + (0u - matrix_fragment_52);

    matrix_fragment_34 =
        uint32_t(combine_words(matrix_fragment_34, 0u) << 0x6u >> 32) + matrix_fragment_55;

    matrix_fragment_12 = matrix_fragment_37 + (0u - matrix_fragment_38);

    matrix_fragment_38 = matrix_fragment_34 & 0xffc0u;

    matrix_fragment_34 = permute_signed(matrix_fragment_13, 0u, 0x8880u);

    matrix_fragment_35 = matrix_fragment_35 + matrix_fragment_36;

    matrix_accumulator_115 = pack_half(packed_half_30, matrix_accumulator_115);

    packed_half_30 = permute_signed(matrix_fragment_12, 0u, 0x8880u);

    matrix_fragment_34 = permute_signed(matrix_fragment_34, 0u, 0x7710u);

    matrix_fragment_35 = matrix_fragment_35 & 0xffffu;

    matrix_fragment_54 = permute_signed(matrix_fragment_54, 0u, 0x7710u);

    packed_half_30 = permute_signed(packed_half_30, 0u, 0x7710u);

    matrix_fragment_34 = uint32_t(uint32_t(matrix_fragment_34) >> 0xau);

    matrix_fragment_52 = permute_signed(matrix_fragment_35, 0u, 0x8880u);

    packed_half_30 = uint32_t(uint32_t(packed_half_30) >> 0xau);

    matrix_fragment_53 = matrix_fragment_53 + matrix_fragment_54;

    matrix_fragment_36 = matrix_fragment_34 & 0x1fu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_52), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    matrix_fragment_37 = packed_half_30 & 0x1fu;

    matrix_fragment_53 = matrix_fragment_53 & 0xffffu;

    matrix_fragment_36 = matrix_fragment_13 + matrix_fragment_36;

    matrix_fragment_34 = matrix_fragment_12 + matrix_fragment_37;

    matrix_fragment_54 = permute_signed(matrix_fragment_53, 0u, 0x8880u);

    matrix_fragment_37 = matrix_fragment_14 & 0xffffu;

    matrix_fragment_14 = matrix_fragment_36 & 0xe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_54), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    matrix_fragment_38 = matrix_fragment_55 + (0u - matrix_fragment_38);

    matrix_fragment_52 = matrix_fragment_34 & 0xe0u;

    matrix_fragment_55 = 0u - matrix_fragment_14;

    matrix_fragment_14 = transpose_half_tile(matrix_accumulator_2);

    matrix_fragment_15 = matrix_fragment_15 & 0xffffu;

    matrix_fragment_37 = uint32_t(uint32_t(matrix_fragment_37) >> 0x5u);

    matrix_fragment_54 = 0u - matrix_fragment_52;

    matrix_fragment_15 = uint32_t(uint32_t(matrix_fragment_15) >> 0x5u);

    condition_1 = (matrix_fragment_37 & 0x1u) != 0;

    matrix_fragment_37 = permute_signed(matrix_fragment_54, 0u, 0x7710u);

    packed_half_30 = transpose_half_tile(matrix_accumulator_109);

    condition_0 = (matrix_fragment_15 & 0x1u) != 0;

    matrix_fragment_54 = permute_signed(matrix_fragment_35, matrix_fragment_54, 0x7632u);

    matrix_fragment_53 = uint32_t(int64_t(int32_t(index_word_156)) * (1ll << 0x1eu) >> 32) + 0x28u;

    matrix_accumulator_109 = condition_0 ? matrix_fragment_54 : matrix_fragment_35;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = transpose_half_tile(matrix_accumulator_108);

    matrix_fragment_68 = uint32_t(int32_t(matrix_fragment_53) >> 0x1fu);

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_accumulator_108 = permute_signed(matrix_fragment_39, matrix_accumulator_108, 0x7632u);

    matrix_fragment_68 =
        uint32_t(combine_words(matrix_fragment_68, 0u) << 0x6u >> 32) + matrix_fragment_53;

    matrix_accumulator_31 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 = condition_1 ? matrix_accumulator_108 : matrix_fragment_39;

    matrix_fragment_52 = matrix_fragment_68 & 0xffc0u;

    matrix_fragment_68 = permute_signed(matrix_fragment_55, 0u, 0x7710u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_accumulator_31 = permute_signed(matrix_fragment_38, 0u, 0x8880u);

    matrix_fragment_52 = matrix_fragment_53 + (0u - matrix_fragment_52);

    matrix_fragment_53 = matrix_fragment_13 + matrix_fragment_68;

    matrix_accumulator_31 = permute_signed(matrix_accumulator_31, 0u, 0x7710u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = matrix_fragment_12 + matrix_fragment_37;

    packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = uint32_t(uint32_t(matrix_accumulator_31) >> 0xau);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(packed_half_30));

    packed_half_30 = matrix_fragment_37 & 0xffffu;

    matrix_fragment_37 = transpose_half_tile(matrix_fragment_6);

    matrix_fragment_35 = matrix_fragment_32 & 0x1fu;

    matrix_fragment_53 = matrix_fragment_53 & 0xffffu;

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_fragment_54 = matrix_fragment_38 + matrix_fragment_35;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = permute_signed(matrix_fragment_53, 0u, 0x8880u);

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_fragment_56 = permute_signed(packed_half_30, 0u, 0x8880u);

    matrix_fragment_53 = matrix_fragment_54 & 0xe0u;

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 = transpose_half_tile(matrix_accumulator_9);

    matrix_fragment_53 = 0u - matrix_fragment_53;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_92));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_56), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    matrix_fragment_39 = transpose_half_tile(matrix_accumulator_8);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_92 = result_word;
    }

    matrix_accumulator_8 = permute_signed(matrix_fragment_52, 0u, 0x8880u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = permute_signed(matrix_fragment_53, 0u, 0x7710u);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = matrix_fragment_38 + matrix_fragment_7;

    matrix_fragment_7 = permute_signed(matrix_accumulator_8, 0u, 0x7710u);

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_fragment_5 = matrix_fragment_34 & 0xffffu;

    matrix_fragment_7 = uint32_t(uint32_t(matrix_fragment_7) >> 0xau);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_fragment_5 = uint32_t(uint32_t(matrix_fragment_5) >> 0x5u);

    matrix_fragment_55 = matrix_fragment_36 & 0xffffu;

    matrix_fragment_7 = matrix_fragment_7 & 0x1fu;

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(0u));

    condition_0 = (matrix_fragment_5 & 0x1u) != 0;

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = uint32_t(uint32_t(matrix_fragment_55) >> 0x5u);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_55 = matrix_fragment_52 + matrix_fragment_7;

    matrix_fragment_32 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_36 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_93 = permute_signed(matrix_fragment_56, matrix_fragment_93, 0x7632u);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_7 = matrix_fragment_55 & 0xe0u;

    matrix_fragment_93 = condition_0 ? matrix_fragment_93 : matrix_fragment_56;

    multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_2, matrix_accumulator_3, 0u, 0u);

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_53 = uint32_t(int64_t(int32_t(index_word_156)) * (1ll << 0x1eu) >> 32) + 0x30u;

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_15 = 0u - matrix_fragment_7;

    matrix_fragment_15 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    matrix_fragment_14 = matrix_fragment_6 & 0xffffu;

    matrix_fragment_52 = matrix_fragment_52 + matrix_fragment_15;

    matrix_fragment_56 = permute_signed(matrix_fragment_14, 0u, 0x8880u);

    matrix_fragment_58 = uint32_t(int32_t(matrix_fragment_53) >> 0x1fu);

    matrix_fragment_52 = matrix_fragment_52 & 0xffffu;

    index_word_156 = uint32_t(int64_t(int32_t(index_word_156)) * (1ll << 0x1eu) >> 32) + 0x38u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_56), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    matrix_fragment_58 =
        uint32_t(combine_words(matrix_fragment_58, 0u) << 0x6u >> 32) + matrix_fragment_53;

    matrix_fragment_59 = permute_signed(matrix_fragment_52, 0u, 0x8880u);

    matrix_fragment_57 = uint32_t(int32_t(index_word_156) >> 0x1fu);

    matrix_fragment_52 = matrix_fragment_58 & 0xffc0u;

    matrix_fragment_58 = matrix_fragment_59;

    matrix_fragment_57 =
        uint32_t(combine_words(matrix_fragment_57, 0u) << 0x6u >> 32) + index_word_156;

    matrix_fragment_59 = matrix_fragment_53 + (0u - matrix_fragment_52);

    matrix_fragment_57 = matrix_fragment_57 & 0xffc0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_58), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_58 = result_word;
    }

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_54 = matrix_fragment_54 & 0xffffu;

    index_word_156 = index_word_156 + (0u - matrix_fragment_57);

    matrix_fragment_57 = permute_signed(matrix_fragment_59, 0u, 0x8880u);

    condition_1 = (matrix_fragment_5 & 0x1u) != 0;

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_5 = permute_signed(matrix_fragment_92, matrix_fragment_5, 0x7632u);

    matrix_fragment_57 = permute_signed(matrix_fragment_57, 0u, 0x7710u);

    matrix_fragment_54 = uint32_t(uint32_t(matrix_fragment_54) >> 0x5u);

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = uint32_t(uint32_t(matrix_fragment_57) >> 0xau);

    matrix_fragment_92 = condition_1 ? matrix_fragment_5 : matrix_fragment_92;

    condition_0 = (matrix_fragment_54 & 0x1u) != 0;

    matrix_fragment_57 = permute_signed(matrix_fragment_56, matrix_fragment_57, 0x7632u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                      select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_85 = matrix_fragment_55 & 0xffffu;

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_84 = condition_0 ? matrix_fragment_57 : matrix_fragment_56;

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_56 = uint32_t(uint32_t(matrix_fragment_85) >> 0x5u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_68 = matrix_fragment_68 & 0x1fu;

    condition_0 = (matrix_fragment_56 & 0x1u) != 0;

    matrix_fragment_57 = permute_signed(matrix_fragment_58, matrix_fragment_57, 0x7632u);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_86 = matrix_fragment_59 + matrix_fragment_68;

    matrix_fragment_78 = condition_0 ? matrix_fragment_57 : matrix_fragment_58;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = matrix_fragment_86 & 0xe0u;

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, 0u, 0u);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    matrix_fragment_79 = 0u - matrix_fragment_68;

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = permute_signed(matrix_fragment_79, 0u, 0x7710u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_68 = matrix_fragment_59 + matrix_fragment_58;

    matrix_fragment_68 = matrix_fragment_68 & 0xffffu;

    matrix_fragment_94 = permute_signed(matrix_fragment_68, 0u, 0x8880u);

    matrix_fragment_68 = index_word_155 + 0x8u;

    index_word_155 = index_word_155 + 0xcu;

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_75 = matrix_fragment_68 + index_word_0 + (0u - index_word_157);

    matrix_fragment_77 = index_word_155 + index_word_0 + (0u - index_word_157);

    split_address_words(packed_half_152, packed_half_153,
                        uint64_t(uint32_t(packed_half_152)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    split_address_words(matrix_fragment_74, matrix_fragment_75,
                        uint64_t(uint32_t(matrix_fragment_75)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    packed_half_152 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_152, packed_half_153) + 0xa8240ull, memory_ranges, error_counts);

    split_address_words(matrix_fragment_76, matrix_fragment_77,
                        uint64_t(uint32_t(matrix_fragment_77)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_79 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_74, matrix_fragment_75) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_fragment_76 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_76, matrix_fragment_77) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_fragment_69 = permute_signed(index_word_156, 0u, 0x8880u);

    matrix_fragment_69 = permute_signed(matrix_fragment_69, 0u, 0x7710u);

    matrix_fragment_69 = uint32_t(uint32_t(matrix_fragment_69) >> 0xau);

    matrix_fragment_69 = matrix_fragment_69 & 0x1fu;

    matrix_fragment_85 = index_word_156 + matrix_fragment_69;

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_69 = matrix_fragment_85 & 0xe0u;

    matrix_fragment_87 = 0u - matrix_fragment_69;

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_69 = permute_signed(matrix_fragment_87, 0u, 0x7710u);

    index_word_156 = index_word_156 + matrix_fragment_69;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_94), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_94 = result_word;
    }

    index_word_156 = index_word_156 & 0xffffu;

    matrix_fragment_96 = permute_signed(index_word_156, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(matrix_fragment_96), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_96 = result_word;
    }

    matrix_fragment_86 = matrix_fragment_86 & 0xffffu;

    matrix_fragment_87 = matrix_fragment_85 & 0xffffu;

    matrix_fragment_85 = uint32_t(uint32_t(matrix_fragment_86) >> 0x5u);

    matrix_fragment_86 = uint32_t(uint32_t(matrix_fragment_87) >> 0x5u);

    condition_0 = (matrix_fragment_85 & 0x1u) != 0;

    matrix_fragment_85 = permute_signed(matrix_fragment_94, matrix_fragment_85, 0x7632u);

    condition_1 = (matrix_fragment_86 & 0x1u) != 0;

    matrix_fragment_86 = condition_0 ? matrix_fragment_85 : matrix_fragment_94;

    matrix_accumulator_19 = transpose_half_tile(matrix_accumulator_19);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, 0u, 0u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_75 = permute_signed(matrix_fragment_96, matrix_fragment_75, 0x7632u);

    matrix_accumulator_18 = transpose_half_tile(matrix_accumulator_18);

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_85 = condition_1 ? matrix_fragment_75 : matrix_fragment_96;

    matrix_accumulator_21 = transpose_half_tile(matrix_accumulator_21);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_45 = transpose_half_tile(matrix_fragment_45);

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_accumulator_17 = transpose_half_tile(matrix_fragment_44);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_accumulator_16 = transpose_half_tile(matrix_accumulator_20);

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_47 = transpose_half_tile(matrix_fragment_47);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                                       select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 = transpose_half_tile(matrix_fragment_46);

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = transpose_half_tile(matrix_fragment_43);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, 0u, 0u);

    matrix_accumulator_23 = transpose_half_tile(matrix_accumulator_23);

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_25 = transpose_half_tile(matrix_accumulator_25);

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_49 = transpose_half_tile(matrix_fragment_49);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_42 = transpose_half_tile(matrix_fragment_42);

    matrix_accumulator_20 = transpose_half_tile(matrix_accumulator_24);

    matrix_accumulator_21 = transpose_half_tile(matrix_fragment_48);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                      select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_accumulator_18 = transpose_half_tile(matrix_accumulator_22);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<0, false, false>(matrix_accumulator_109));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<0, false, false>(matrix_accumulator_108));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_25 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                      select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    multiply_accumulate_half(packed_half_30, matrix_accumulator_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, packed_half_30,
                             matrix_accumulator_31);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_accumulator_8,
                             matrix_accumulator_9);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<0, false, false>(matrix_fragment_78));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<0, false, false>(matrix_fragment_86));

    matrix_accumulator_110 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_58,
                             matrix_fragment_59);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<0, false, false>(matrix_fragment_85));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_114));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                      select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                          select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                      select_half_operand<2, false, false>(matrix_fragment_12));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_30),
                                          select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_36,
                             matrix_fragment_37);

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_144, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_accumulator_17));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x1000u, matrix_accumulator_8, matrix_accumulator_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_accumulator_16));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144 + 0x3000u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x2000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(matrix_accumulator_24));

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_accumulator_25));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                       select_half_operand<2, false, false>(matrix_accumulator_24));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_accumulator_26));

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_144, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x1000u, matrix_fragment_52,
                                         matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x2000u, matrix_fragment_40,
                                         matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_144 + 0x3000u, matrix_fragment_32,
                                         matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_5);

    matrix_accumulator_31 = unpack_fp8_bits(matrix_fragment_11);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_16);

    split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                        uint64_t(uint32_t(thread_row_158)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_9 >> 16);

    matrix_fragment_81 = unpack_fp8_bits(matrix_fragment_4);

    matrix_accumulator_24 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_accumulator_25 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_accumulator_17);

    matrix_accumulator_29 = unpack_fp8_bits(matrix_fragment_7);

    matrix_accumulator_20 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_fragment_10);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    matrix_accumulator_28 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_accumulator_21 = unpack_fp8_bits(matrix_accumulator_8 >> 16);

    matrix_fragment_7 = unpack_fp8_bits(matrix_accumulator_9);

    matrix_accumulator_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_162));

    packed_half_30 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_78 = unpack_fp8_bits(matrix_accumulator_18);

    matrix_fragment_37 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_6 = unpack_fp8_bits(matrix_accumulator_8);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_10 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

    matrix_accumulator_18 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    matrix_accumulator_19 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(packed_half_162));

    split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                        combine_words(matrix_accumulator_26, matrix_accumulator_27) + 0x98440u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_6 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_7 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_5 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_24),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_accumulator_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_25),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_30),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    uniform_index_word_4 = 0u;

    __syncthreads();
    do {
        split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_28, matrix_accumulator_29) - 0x200ull,
            matrix_fragment_32, matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_28, matrix_accumulator_29), matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        packed_half_30 = (index_word_0 << 0x4u) + uniform_index_word_5;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_6 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                            combine_words(matrix_accumulator_26, matrix_accumulator_27) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_30, matrix_fragment_48, matrix_fragment_49, matrix_fragment_50,
            matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_5 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_30 + 0x1000u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_6;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_30 + 0x2000u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_30 + 0x3000u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_8, matrix_accumulator_9, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_8,
                                 matrix_accumulator_9);

        multiply_accumulate_half(matrix_accumulator_24, matrix_accumulator_25, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, matrix_accumulator_24,
                                 matrix_accumulator_25);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_32, matrix_fragment_33, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_78,
                                 matrix_fragment_79);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_18,
                                 matrix_accumulator_19);
    } while (!uniform_condition_0);
    matrix_accumulator_28 = blockIdx.y;

    matrix_accumulator_29 = parameters[parameter_geometry_or_control_11];

    condition_1 = int32_t(index_word_150) >= int32_t(index_word_146);

    condition_4 = int32_t(index_word_151) != int32_t(0u);

    matrix_fragment_33 = threadIdx.x;

    matrix_accumulator_31 = index_word_150 * index_word_145 + index_word_145;

    index_word_145 = index_word_150 * index_word_145 + index_word_148;

    matrix_fragment_32 = threadIdx.z;

    matrix_accumulator_26 = parameters[parameter_output_low_2];
    matrix_accumulator_27 = parameters[parameter_output_high_3];

    packed_half_30 = matrix_accumulator_28 * 0x8u + matrix_accumulator_29;

    matrix_accumulator_29 = index_word_150 + 0x1u;

    matrix_accumulator_28 = thread_row_158 * 0x80u;

    condition_1 = int32_t(packed_half_30) > int32_t(0u - 0x4u) && !condition_1;

    condition_0 = int32_t(matrix_accumulator_29) >= int32_t(index_word_146);

    matrix_accumulator_29 = index_word_148 + matrix_accumulator_31;

    index_word_145 = index_word_145 * 0x400u + matrix_accumulator_28;

    condition_2 = (matrix_fragment_32 | matrix_fragment_33 | thread_row_158) != 0;

    condition_3 = condition_1 & condition_4;

    matrix_accumulator_29 = matrix_accumulator_29 * 0x400u + matrix_accumulator_28;

    condition_0 = int32_t(packed_half_30) > int32_t(0u - 0x8u) && !condition_0;

    split_address_words(matrix_fragment_32, matrix_fragment_33,
                        int64_t(int32_t(matrix_accumulator_29)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_26, matrix_accumulator_27));

    split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                        int64_t(int32_t(index_word_145)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_26, matrix_accumulator_27));
    if (condition_3) {
        matrix_accumulator_28 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_25),
                          select_half_operand<2, false, false>(0u));

        packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_25 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_24),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_29 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_9),
                          select_half_operand<2, false, false>(matrix_accumulator_28));

        matrix_accumulator_31 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                          select_half_operand<2, false, false>(packed_half_30));

        matrix_accumulator_28 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_8),
                          select_half_operand<2, false, false>(matrix_accumulator_25));

        split_address_words(matrix_accumulator_8, matrix_accumulator_9,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));

        packed_half_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_8, matrix_accumulator_9), matrix_accumulator_28,
            matrix_accumulator_29, packed_half_30, matrix_accumulator_31, memory_ranges,
            error_counts);
    }
    condition_3 = int32_t(index_word_149) != int32_t(0u);

    matrix_accumulator_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                          select_half_operand<2, false, false>(0u));

    condition_1 = condition_1 & condition_3;

    matrix_accumulator_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));
    if (condition_1) {
        matrix_accumulator_8 = permute_signed(matrix_accumulator_8, matrix_fragment_43, 0x5410u);

        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_26, matrix_accumulator_27));

        matrix_accumulator_9 = permute_signed(matrix_accumulator_9, matrix_fragment_42, 0x5410u);

        matrix_accumulator_25 = permute_signed(matrix_accumulator_25, matrix_fragment_14, 0x5410u);

        matrix_accumulator_24 =
            permute_signed(matrix_accumulator_24, matrix_accumulator_29, 0x5410u);

        matrix_accumulator_26 = matrix_accumulator_9;

        matrix_accumulator_27 = matrix_accumulator_8;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_6, matrix_fragment_7) + 0x1000ull, matrix_accumulator_24,
            matrix_accumulator_25, matrix_accumulator_26, matrix_accumulator_27, memory_ranges,
            error_counts);
    }
    condition_1 = condition_0 & condition_4;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                         select_half_operand<2, false, false>(0u));

    condition_0 = condition_0 & condition_3;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(0u));
    if (condition_1) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_4, 0x5410u);

        matrix_fragment_14 = permute_signed(matrix_fragment_14, matrix_fragment_5, 0x5410u);

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_32, matrix_fragment_33));

        matrix_fragment_13 = permute_signed(matrix_accumulator_8, matrix_fragment_13, 0x5410u);

        matrix_fragment_12 = permute_signed(matrix_fragment_6, matrix_fragment_7, 0x5410u);

        store_four_words_streaming<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                                matrix_fragment_12, matrix_fragment_13,
                                                matrix_fragment_14, matrix_fragment_15,
                                                memory_ranges, error_counts);
    }
    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_9 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_8 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(0u));
    if (condition_0) {
        split_address_words(matrix_fragment_32, matrix_fragment_33,
                            int64_t(int32_t(index_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_32, matrix_fragment_33));

        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_accumulator_8, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_accumulator_9, 0x5410u);

        matrix_fragment_5 = permute_signed(matrix_fragment_5, matrix_accumulator_2, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_accumulator_3, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_32, matrix_fragment_33) + 0x1000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_2) {
        return;
    }

    uniform_index_word_4 = blockIdx.x;

    matrix_accumulator_2 = parameters[parameter_publication_optional_low_16];
    matrix_accumulator_3 = parameters[parameter_publication_optional_high_17];

    address_word_147 = address_word_147 + uniform_index_word_4;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(address_word_147)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__ __maxnreg__(184) void phoenix_window_attention_channels_256_attention(
    KernelArguments parameters) {
    run_window_attention_channels_256_attention<false>(parameters.words, nullptr, nullptr);
}
