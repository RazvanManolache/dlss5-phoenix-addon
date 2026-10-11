// DLSS5 Phoenix optimized kernel: window_attention/channels_64/attention.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 5120
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
    parameter_scalar_control_10 = 10,
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
__device__ __forceinline__ void run_window_attention_channels_64_attention(
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
    uint32_t matrix_accumulator_16 = 0;
    uint32_t matrix_accumulator_17 = 0;
    uint32_t matrix_accumulator_18 = 0;
    uint32_t matrix_accumulator_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_accumulator_22 = 0;
    uint32_t matrix_accumulator_23 = 0;
    uint32_t matrix_fragment_24 = 0;
    uint32_t matrix_fragment_25 = 0;
    uint32_t matrix_fragment_26 = 0;
    uint32_t matrix_fragment_27 = 0;
    uint32_t matrix_accumulator_28 = 0;
    uint32_t matrix_accumulator_29 = 0;
    uint32_t matrix_accumulator_30 = 0;
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
    uint32_t matrix_accumulator_48 = 0;
    uint32_t matrix_accumulator_49 = 0;
    uint32_t matrix_accumulator_50 = 0;
    uint32_t matrix_accumulator_51 = 0;
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
    uint32_t matrix_accumulator_96 = 0;
    uint32_t matrix_accumulator_97 = 0;
    uint32_t matrix_accumulator_98 = 0;
    uint32_t matrix_accumulator_99 = 0;
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
    uint32_t matrix_fragment_128 = 0;
    uint32_t matrix_fragment_129 = 0;
    uint32_t matrix_accumulator_130 = 0;
    uint32_t matrix_accumulator_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t matrix_accumulator_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t packed_half_138 = 0;
    uint32_t packed_half_139 = 0;
    uint32_t matrix_accumulator_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t packed_half_142 = 0;
    uint32_t packed_half_143 = 0;
    uint32_t packed_half_144 = 0;
    uint32_t packed_half_145 = 0;
    uint32_t packed_half_146 = 0;
    uint32_t packed_half_147 = 0;
    uint32_t packed_half_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t packed_half_150 = 0;
    uint32_t index_word_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t index_word_155 = 0;
    uint32_t index_word_156 = 0;
    uint32_t index_word_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t address_word_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t reused_word_162 = 0;
    uint32_t index_word_163 = 0;
    uint32_t index_word_164 = 0;
    uint32_t thread_row_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
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

    matrix_fragment_10 = blockIdx.y;

    uniform_parameter_word_8 = parameters[parameter_geometry_or_control_15];

    matrix_fragment_4 = parameters[parameter_scalar_control_10];
    matrix_fragment_5 = parameters[parameter_geometry_or_control_11];

    matrix_fragment_7 = threadIdx.x;

    uniform_index_word_5 = parameters[parameter_tensor_height_8];

    reused_word_162 = 0u;

    matrix_fragment_12 = threadIdx.y;

    uniform_parameter_word_7 = parameters[parameter_geometry_or_control_14];

    matrix_fragment_9 = parameters[parameter_tensor_width_9];

    matrix_fragment_8 = threadIdx.z;

    packed_half_3 = blockIdx.x;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_8));

    uniform_index_word_4 = 0u - uniform_index_word_4 + uniform_index_word_5;

    packed_half_2 = min(int32_t(0u), int32_t(matrix_fragment_4));

    uniform_condition_0 = uint32_t(uniform_index_word_4) != 0u;

    uniform_index_word_6 = min(int32_t(0u), int32_t(uniform_parameter_word_7));

    packed_half_0 = matrix_fragment_10 * 0x8u + matrix_fragment_5;

    matrix_fragment_5 = 0u - packed_half_2 + matrix_fragment_9;

    packed_half_2 = matrix_fragment_9 + (0u - uniform_index_word_6);

    matrix_fragment_6 = packed_half_0 + (0u - uniform_parameter_word_8);

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_4 + 0xffffffffu;
    }

    condition_4 = uint32_t(matrix_fragment_5) != 0u;

    condition_3 = uint32_t(packed_half_2) != 0u;

    uniform_index_word_4 = 0u;

    matrix_fragment_11 = matrix_fragment_6 + 0x8u;

    if (uniform_condition_0) {
        uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_5) >> 0x1fu);
    }

    condition_0 = (matrix_fragment_12 | matrix_fragment_7 | matrix_fragment_8) != 0;

    uniform_parameter_word_8 = 0u;
    uniform_reused_word_9 = 0u;

    matrix_fragment_8 = matrix_fragment_6 + 0x7u;

    condition_1 = int32_t(matrix_fragment_11) > int32_t(0u);

    matrix_fragment_4 = packed_half_3 * 0x8u + matrix_fragment_4;

    if (uniform_condition_0) {
        uniform_index_word_6 =
            uint32_t(combine_words(uniform_index_word_6, 0u) << 0x3u >> 32) + uniform_index_word_5;
    }

    matrix_fragment_6 = max(int32_t(0u), int32_t(matrix_fragment_6));

    matrix_fragment_8 = uint32_t(combine_words(matrix_fragment_8, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_fragment_9 = matrix_fragment_4 + (0u - uniform_parameter_word_7);

    packed_half_3 = uint32_t(int32_t(packed_half_0) >> 0x1fu);

    matrix_fragment_8 = condition_1 ? matrix_fragment_8 : 0u;

    matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_6) >> 0x3u);

    matrix_fragment_7 = matrix_fragment_9 + 0x8u;

    packed_half_3 = uint32_t(combine_words(packed_half_3, 0u) << 0x2u >> 32) + packed_half_0;

    matrix_fragment_11 = min(int32_t(matrix_fragment_8), int32_t(uniform_index_word_4));

    if (condition_4) {
        packed_half_0 = matrix_fragment_5 + 0xffffffffu;
    }

    if (condition_3) {
        packed_half_2 = packed_half_2 + 0xffffffffu;
    }

    condition_2 = int32_t(matrix_fragment_7) > int32_t(0u);

    matrix_fragment_8 = matrix_fragment_9 + 0x7u;

    condition_0 = int32_t(matrix_fragment_6) >= int32_t(matrix_fragment_11) || condition_0;

    if (condition_4) {
        matrix_fragment_7 = uint32_t(int32_t(packed_half_0) >> 0x1fu);
    }

    if (condition_3) {
        matrix_fragment_5 = uint32_t(int32_t(packed_half_2) >> 0x1fu);
    }

    if (condition_4) {
        matrix_fragment_7 =
            uint32_t(combine_words(matrix_fragment_7, 0u) << 0x3u >> 32) + packed_half_0;
    }

    if (condition_3) {
        packed_half_2 =
            uint32_t(combine_words(matrix_fragment_5, 0u) << 0x3u >> 32) + packed_half_2;
    }

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    if (condition_4) {
        matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_7) >> 0x3u);
    }

    matrix_fragment_8 = uint32_t(combine_words(matrix_fragment_8, 0u) << 0x1du >> 32) + 0x1u;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_4;

    if (condition_4) {
        reused_word_162 = matrix_fragment_7 * matrix_fragment_10 + matrix_fragment_10;
    }

    packed_half_0 = 0u;

    index_word_151 = uint32_t(int32_t(packed_half_3) >> 0x2u);

    if (condition_3) {
        packed_half_0 = uint32_t(int64_t(int32_t(packed_half_2)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_fragment_9 = max(int32_t(0u), int32_t(matrix_fragment_9));

    index_word_156 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    packed_half_3 = condition_2 ? matrix_fragment_8 : 0u;
    if (!condition_0) {
        matrix_fragment_9 = uint32_t(uint32_t(matrix_fragment_9) >> 0x3u);

        matrix_fragment_8 = min(int32_t(packed_half_3), int32_t(packed_half_0));
        do {
            condition_0 = int32_t(matrix_fragment_9) >= int32_t(matrix_fragment_8);
            if (!condition_0) {
                matrix_fragment_10 = matrix_fragment_6 * packed_half_0;

                matrix_fragment_5 = matrix_fragment_9;
                do {
                    packed_half_2 = parameters[parameter_wait_optional_low_12];
                    packed_half_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_7 = matrix_fragment_10 + matrix_fragment_5;

                    split_address_words(packed_half_2, packed_half_3,
                                        int64_t(int32_t(matrix_fragment_7)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(packed_half_2, packed_half_3));

                    matrix_fragment_7 = pipeline_read<CheckBounds>(
                        combine_words(packed_half_2, packed_half_3), memory_ranges, error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_8);

                    condition_0 = int32_t(matrix_fragment_7) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_7 = pipeline_read<CheckBounds>(
                                combine_words(packed_half_2, packed_half_3), memory_ranges,
                                error_counts);

                            condition_0 = int32_t(matrix_fragment_7) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            matrix_fragment_6 = matrix_fragment_6 + 0x1u;

            condition_0 = uint32_t(matrix_fragment_6) != uint32_t(matrix_fragment_11);
        } while (condition_0);
    }
    matrix_fragment_6 = parameters[parameter_tensor_height_8];
    matrix_fragment_7 = parameters[parameter_tensor_width_9];

    packed_half_2 = index_word_151 + matrix_fragment_12;

    condition_4 = false;

    address_word_159 = threadIdx.x & 31u;

    condition_6 = false;

    __syncwarp(__activemask());

    condition_2 = false;

    address_word_160 = parameters[parameter_trained_weights_low_4];
    address_word_161 = parameters[parameter_trained_weights_high_5];

    condition_3 = false;

    matrix_fragment_5 = 0u;

    matrix_fragment_11 = 0u;

    matrix_fragment_13 = 0u;

    matrix_fragment_10 = index_word_156 + 0x1u;

    packed_half_0 = matrix_fragment_6 & 0xfffffffcu;

    packed_half_3 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    __syncthreads();

    condition_5 = uint32_t(packed_half_0) != uint32_t(0x4u);

    packed_half_3 = uint32_t(combine_words(packed_half_3, 0u) << 0x2u >> 32) + matrix_fragment_6;

    packed_half_0 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

    index_word_157 = uint32_t(int32_t(packed_half_3) >> 0x2u);

    packed_half_0 = uint32_t(combine_words(packed_half_0, 0u) << 0x2u >> 32) + matrix_fragment_7;

    matrix_fragment_6 = 0u;

    index_word_155 = uint32_t(int32_t(packed_half_0) >> 0x2u);

    if (condition_5) {
        condition_1 = int32_t(packed_half_2) >= int32_t(index_word_157);

        condition_0 = int32_t(packed_half_2) >= int32_t(index_word_157);

        matrix_fragment_5 = index_word_155 * packed_half_2;

        condition_4 = int32_t(packed_half_2) < int32_t(0u) || condition_1;
        condition_1 = int32_t(packed_half_2) >= int32_t(index_word_157);

        condition_6 = int32_t(packed_half_2) < int32_t(0u) || condition_0;

        condition_0 = int32_t(packed_half_2) >= int32_t(index_word_157);

        matrix_fragment_6 = index_word_155 * packed_half_2;
    }

    packed_half_0 = uint32_t(condition_4) << 4;

    if (condition_5) {
        matrix_fragment_11 = index_word_155 * packed_half_2;

        condition_2 = int32_t(packed_half_2) < int32_t(0u) || condition_1;

        condition_3 = int32_t(packed_half_2) < int32_t(0u) || condition_0;
    }

    condition_0 = true;

    if (condition_5) {
        matrix_fragment_13 = index_word_155 * packed_half_2;

        condition_0 = !condition_4;
    }

    condition_4 = true;

    if (condition_5) {
        condition_4 = !condition_2;
    }

    condition_1 = true;

    packed_half_2 = uint32_t(condition_4) << 4;

    condition_4 = true;

    if (condition_5) {
        condition_4 = !condition_3;

        condition_1 = !condition_6;
    }

    matrix_fragment_8 = uint32_t(condition_4) << 4;

    condition_4 = int32_t(packed_half_0) != int32_t(0u);

    packed_half_0 = matrix_fragment_7 & 0xfffffffcu;

    condition_5 = int32_t(index_word_156) >= int32_t(index_word_155);

    matrix_fragment_7 = condition_6 ? index_word_156 : 0u;

    packed_half_3 = condition_4 ? index_word_156 : 0u;

    condition_4 = uint32_t(packed_half_0) == uint32_t(0x4u) || condition_4;

    condition_6 = uint32_t(packed_half_0) == uint32_t(0x4u) || condition_6;

    condition_5 = int32_t(matrix_fragment_4) > int32_t(0u - 0x4u) && !condition_5;

    matrix_fragment_9 = condition_2 ? matrix_fragment_10 : 0u;

    condition_2 = uint32_t(packed_half_0) == uint32_t(0x4u) || condition_2;

    condition_0 = condition_0 & (condition_4 | condition_5);

    condition_4 = int32_t(matrix_fragment_10) >= int32_t(index_word_155);

    condition_1 = condition_1 & (condition_6 | condition_5);

    condition_6 = int32_t(packed_half_2) != int32_t(0u);

    condition_4 = int32_t(matrix_fragment_4) > int32_t(0u - 0x8u) && !condition_4;

    condition_2 = condition_6 & (condition_2 | condition_4);

    condition_6 = uint32_t(packed_half_0) != uint32_t(0x4u);

    if (!condition_0) {
        matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_12 = !condition_6 ? matrix_fragment_9 : matrix_fragment_10;

    matrix_fragment_4 = !condition_6 ? packed_half_3 : index_word_156;

    matrix_fragment_7 = !condition_6 ? matrix_fragment_7 : index_word_156;

    if (condition_0) {
        packed_half_2 = parameters[parameter_input_low_0];
        packed_half_3 = parameters[parameter_input_high_1];
    }

    if (!condition_0) {
        matrix_accumulator_19 =
            permute_signed(matrix_accumulator_19, matrix_accumulator_19, 0x5410u);
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_12 + matrix_fragment_11;
    }

    if (condition_1) {
        matrix_fragment_6 = matrix_fragment_7 + matrix_fragment_6;
    }

    if (!condition_6) {
        matrix_fragment_10 = condition_3 ? matrix_fragment_10 : 0u;
    }

    condition_6 = int32_t(matrix_fragment_8) != int32_t(0u);

    condition_3 = uint32_t(packed_half_0) == uint32_t(0x4u) || condition_3;

    if (condition_2) {
        matrix_fragment_14 = parameters[parameter_input_low_0];
        matrix_fragment_15 = parameters[parameter_input_high_1];

        matrix_fragment_11 = matrix_fragment_11 * 0x100u;
    }

    if (condition_0) {
        matrix_fragment_5 = matrix_fragment_4 + matrix_fragment_5;
    }

    if (condition_1) {
        matrix_fragment_7 = matrix_fragment_6 * 0x100u;
    }

    condition_3 = condition_6 & (condition_3 | condition_4);

    if (!condition_1) {
        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                      select_half_operand<2, false, false>(0u));
    }

    if (condition_0) {
        matrix_fragment_5 = matrix_fragment_5 * 0x100u;
    }

    if (condition_1) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];
    }

    index_word_164 = address_word_159;

    if (!condition_0) {
        matrix_accumulator_16 = matrix_accumulator_19;

        matrix_accumulator_17 = matrix_accumulator_19;

        matrix_accumulator_18 = matrix_accumulator_19;
    }

    if (condition_0) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    if (condition_3) {
        matrix_accumulator_22 = parameters[parameter_input_low_0];
        matrix_accumulator_23 = parameters[parameter_input_high_1];

        matrix_fragment_10 = matrix_fragment_10 + matrix_fragment_13;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_2) {
        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        matrix_fragment_25 = matrix_fragment_10 * 0x100u;
    }

    if (!condition_2) {
        matrix_fragment_11 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5410u);
    }

    if (condition_1) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_1) {
        matrix_fragment_7 = permute_signed(packed_half_0, packed_half_0, 0x5410u);
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_3) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (!condition_1) {
        matrix_fragment_4 = matrix_fragment_7;

        matrix_fragment_5 = matrix_fragment_7;

        matrix_fragment_6 = matrix_fragment_7;
    }

    if (condition_1) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_2) {
        matrix_fragment_8 = matrix_fragment_11;

        matrix_fragment_9 = matrix_fragment_11;

        matrix_fragment_10 = matrix_fragment_11;
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_22, matrix_accumulator_23));
    }

    if (!condition_3) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    index_word_163 = index_word_164 & 0xfffffffcu;

    if (condition_0) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    if (!condition_3) {
        matrix_fragment_14 = matrix_fragment_15;
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_20, matrix_accumulator_21), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    if (condition_3) {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_22, matrix_accumulator_23));
    }

    matrix_fragment_27 = address_word_159 + 0xcu + (0u - index_word_163);

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(combine_words(packed_half_2, packed_half_3),
                                                   matrix_accumulator_16, matrix_accumulator_17,
                                                   matrix_accumulator_18, matrix_accumulator_19,
                                                   memory_ranges, error_counts);
    }

    if (!condition_3) {
        matrix_fragment_12 = matrix_fragment_15;

        matrix_fragment_13 = matrix_fragment_15;
    }

    matrix_fragment_25 = address_word_159 + 0x8u + (0u - index_word_163);

    matrix_accumulator_21 = address_word_159 + 0x4u + (0u - index_word_163);

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0x200ull,
            matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
            memory_ranges, error_counts);
    }

    packed_half_3 = 0u - index_word_163 + address_word_159;

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        uint64_t(uint32_t(matrix_accumulator_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_accumulator_31 = address_word_159 + 0x14u + (0u - index_word_163);

    split_address_words(packed_half_2, packed_half_3,
                        int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_33 = address_word_159 + 0x18u + (0u - index_word_163);

    matrix_accumulator_29 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                        uint64_t(uint32_t(matrix_fragment_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_27 = address_word_159 + 0x10u + (0u - index_word_163);

    matrix_accumulator_111 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, packed_half_3) + 0x7010ull, memory_ranges, error_counts);

    matrix_fragment_35 = address_word_159 + 0x1cu + (0u - index_word_163);

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        uint64_t(uint32_t(matrix_fragment_25)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_accumulator_23 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        uint64_t(uint32_t(matrix_accumulator_31)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_accumulator_28 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_24, matrix_fragment_25) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(packed_half_2, packed_half_3,
                        uint64_t(uint32_t(matrix_fragment_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    split_address_words(matrix_fragment_26, matrix_fragment_27,
                        uint64_t(uint32_t(matrix_fragment_33)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_32 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_2, packed_half_3) + 0x7010ull, memory_ranges, error_counts);

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        uint64_t(uint32_t(matrix_fragment_35)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_33 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_fragment_38 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_26, matrix_fragment_27) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_24, matrix_fragment_25) + 0x7010ull, memory_ranges,
        error_counts);

    uniform_condition_0 = true;

    matrix_accumulator_137 = unpack_fp8_bits(matrix_fragment_6);

    matrix_accumulator_136 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_accumulator_135 = unpack_fp8_bits(matrix_fragment_7);

    matrix_accumulator_134 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_129 = unpack_fp8_bits(matrix_fragment_8);

    packed_half_149 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

    packed_half_146 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    packed_half_145 = unpack_fp8_bits(matrix_accumulator_18);

    packed_half_143 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_fragment_128 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_accumulator_114 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_accumulator_113 = unpack_fp8_bits(matrix_fragment_10);

    packed_half_0 = unpack_fp8_bits(matrix_fragment_11);

    matrix_accumulator_141 = unpack_fp8_bits(matrix_fragment_4);

    matrix_accumulator_140 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    packed_half_139 = unpack_fp8_bits(matrix_fragment_5);

    packed_half_138 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_accumulator_115 = unpack_fp8_bits(matrix_fragment_9);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_7 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    packed_half_144 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    packed_half_142 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    matrix_accumulator_112 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_15);

    packed_half_150 = unpack_fp8_bits(matrix_accumulator_16);

    packed_half_148 = unpack_fp8_bits(matrix_accumulator_17);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_128),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_11 = packed_half_0;

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    packed_half_0 = packed_half_2;

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                            select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                            select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_10 = 0u;

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_129),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_accumulator_23));

    packed_half_2 = matrix_fragment_40;

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_111));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_39));
    do {
        matrix_accumulator_103 = matrix_fragment_10 * 0x800u;

        split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                            uint64_t(uint32_t(matrix_accumulator_103)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_160, address_word_161));

        split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_102, matrix_accumulator_103));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103), matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x200ull,
            matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50,
            matrix_accumulator_51, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1000ull,
            matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x400ull,
            matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x600ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1200ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1400ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1600ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        matrix_accumulator_101 = matrix_fragment_10 * 0x400u;

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x800ull,
            matrix_fragment_92, matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
            memory_ranges, error_counts);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_160, address_word_161));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1800ull,
            matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_98,
            matrix_accumulator_99, memory_ranges, error_counts);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4000ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4200ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_59 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_56 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_57 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                          select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_58 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                          select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_129),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                          select_half_operand<2, false, false>(matrix_fragment_80));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                          select_half_operand<2, false, false>(matrix_fragment_82));

        matrix_fragment_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_82 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_120 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                               select_half_operand<2, false, false>(0u));

        matrix_fragment_80 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                          select_half_operand<2, false, false>(matrix_fragment_80));

        matrix_fragment_82 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                          select_half_operand<2, false, false>(matrix_fragment_82));

        packed_half_147 = 0x2b28u;

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_76, matrix_fragment_77, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_76, matrix_fragment_77, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_78, matrix_fragment_79, 0u, 0u);

        matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_138),
                                           select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_78, matrix_fragment_79, 0u, 0u);

        matrix_fragment_81 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_139),
                          select_half_operand<2, false, false>(matrix_fragment_76));

        matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_78 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_83 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(matrix_fragment_78));

        matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_76 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                          select_half_operand<2, false, false>(matrix_fragment_76));

        matrix_fragment_77 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                          select_half_operand<2, false, false>(matrix_fragment_77));

        matrix_fragment_78 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                          select_half_operand<2, false, false>(matrix_fragment_78));

        matrix_fragment_79 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                          select_half_operand<2, false, false>(matrix_accumulator_120));

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_accumulator_50, matrix_accumulator_51, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_accumulator_48, matrix_accumulator_49, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_50, matrix_accumulator_51, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_48, matrix_accumulator_49, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_50,
                                 matrix_accumulator_51);

        multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_54,
                                 matrix_fragment_55);

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_106), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_122,
                                 matrix_accumulator_123);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_72, matrix_fragment_73, matrix_fragment_60,
                                 matrix_fragment_61);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_107), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_48,
                                 matrix_accumulator_49);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_accumulator_117 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_accumulator_116 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_120 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_120),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120),
            select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

        matrix_fragment_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        matrix_fragment_63 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_63),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_124,
                                 matrix_accumulator_125);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x44004400u);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_118), 0x44004400u);

        matrix_fragment_53 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_105), 0x44004400u);

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_119), 0x44004400u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_accumulator_118 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0x44004400u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        matrix_accumulator_119 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_fragment_63 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_63),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_84, matrix_fragment_85, 0u, 0u);

        matrix_fragment_54 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_86, matrix_fragment_87, 0u, 0u);

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        matrix_fragment_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                           select_half_operand<2, false, false>(0u));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4400ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                          select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_86, matrix_fragment_87, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_84, matrix_fragment_85, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                                 matrix_fragment_53);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_52 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_accumulator_130 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_48), 0x44004400u);

        matrix_accumulator_131 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4600ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_54,
                                 matrix_fragment_55);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0xa00ull,
            matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        matrix_fragment_70 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_50), 0x44004400u);

        matrix_fragment_88 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_49), 0x44004400u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_51), 0x44004400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_88),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        packed_half_152 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_45 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_46,
                                 matrix_fragment_47);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1a00ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_132 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

        packed_half_153 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                                select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_accumulator_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_45),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45),
            select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_47 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0x44004400u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_accumulator_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_48),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_48),
            select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_accumulator_123 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73), 0x44004400u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_45),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                select_half_operand<2, false, false>(matrix_accumulator_132));

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45),
            select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        packed_half_154 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_accumulator_127 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        matrix_fragment_70 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4800ull,
            matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4a00ull,
            matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50,
            matrix_accumulator_51, memory_ranges, error_counts);

        matrix_accumulator_116 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_70 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        matrix_fragment_72 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

        matrix_accumulator_127 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                select_half_operand<2, false, false>(matrix_accumulator_127));

        matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                           select_half_operand<2, false, false>(0u));

        packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                        select_half_operand<2, false, false>(0u));

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_72),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        packed_half_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                        select_half_operand<2, false, false>(0u));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_fragment_73), 0x3b283b28u);

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                           select_half_operand<2, false, false>(packed_half_153));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                          select_half_operand<2, false, false>(packed_half_154));

        matrix_accumulator_117 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_accumulator_131 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_73 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_74,
                                 matrix_fragment_75);

        matrix_fragment_72 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                          select_half_operand<2, false, false>(matrix_accumulator_131));

        matrix_fragment_73 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_fragment_75 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_116 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_74 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_75 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(matrix_accumulator_116));

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_84,
                                 matrix_fragment_85);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_104,
                                 matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_86,
                                 matrix_fragment_87);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1c00ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0xc00ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0xe00ull,
            matrix_fragment_72, matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_88, matrix_fragment_89, matrix_fragment_68,
                                 matrix_fragment_69);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_88, matrix_fragment_89, matrix_fragment_66,
                                 matrix_fragment_67);

        matrix_fragment_66 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0x44004400u);

        matrix_fragment_67 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

        matrix_fragment_66 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66), 0xc400c400u);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_64,
                                 matrix_fragment_65);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_66),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_102, matrix_accumulator_103) + 0x1e00ull,
            matrix_fragment_68, matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
            memory_ranges, error_counts);

        matrix_accumulator_126 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

        matrix_accumulator_103 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93), 0x44004400u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0xc400c400u);

        matrix_accumulator_127 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0xc400c400u);

        matrix_accumulator_126 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        matrix_accumulator_126 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x3b283b28u);

        matrix_fragment_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0xc400c400u);

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                select_half_operand<2, false, false>(matrix_accumulator_126));

        matrix_fragment_93 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_127),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_102 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93), 0xc400c400u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_92),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127),
            select_half_operand<2, false, false>(matrix_accumulator_103), 0x3b283b28u);

        matrix_fragment_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92),
            select_half_operand<2, false, false>(matrix_fragment_93), 0x3b283b28u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_102),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_accumulator_103 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_102),
            select_half_operand<2, false, false>(matrix_fragment_92), 0x3b283b28u);

        matrix_fragment_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_accumulator_102 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0xc400c400u);

        matrix_accumulator_126 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_102), 0xc400c400u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_92),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92),
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_126),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_103 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_accumulator_126 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126),
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

        matrix_accumulator_102 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(matrix_fragment_93));

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_92));

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        matrix_accumulator_126 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_accumulator_126));

        matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_103 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                          select_half_operand<2, false, false>(matrix_accumulator_103));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                          select_half_operand<2, false, false>(matrix_fragment_55));

        multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_92,
                                 matrix_fragment_93);

        matrix_fragment_95 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_99), 0x44004400u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4c00ull,
            matrix_fragment_64, matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_48, matrix_accumulator_49,
                                 matrix_accumulator_104, matrix_accumulator_105);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_50, matrix_accumulator_51,
                                 matrix_accumulator_132, matrix_accumulator_133);

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_98), 0x44004400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_91 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_91),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91),
            select_half_operand<2, false, false>(matrix_fragment_94), 0x3b283b28u);

        matrix_fragment_95 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95), 0xc400c400u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4e00ull,
            matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
            memory_ranges, error_counts);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_fragment_90 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_95),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_101 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_90 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_95),
            select_half_operand<2, false, false>(matrix_fragment_90), 0x3b283b28u);

        matrix_accumulator_101 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101), 0xc400c400u);

        matrix_accumulator_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                select_half_operand<2, false, false>(matrix_fragment_94));

        matrix_fragment_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92), 0x44004400u);

        matrix_accumulator_105 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                select_half_operand<2, false, false>(matrix_fragment_90));

        matrix_fragment_90 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_101),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94), 0xc400c400u);

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0xc400c400u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_accumulator_101 = matrix_fragment_10 * 0x200u;

        matrix_fragment_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_94),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_91 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_94),
            select_half_operand<2, false, false>(matrix_fragment_91), 0x3b283b28u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(matrix_fragment_88), 0x3b283b28u);

        matrix_fragment_90 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_93), 0x44004400u);

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_fragment_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0xc400c400u);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_160, address_word_161));

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_92),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_95 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        matrix_fragment_94 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_92),
            select_half_operand<2, false, false>(matrix_fragment_94), 0x3b283b28u);

        matrix_accumulator_104 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(matrix_fragment_95));

        packed_half_152 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_88));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6400ull,
            matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_98,
            matrix_accumulator_99, memory_ranges, error_counts);

        packed_half_153 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                select_half_operand<2, false, false>(matrix_fragment_94));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6000ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6200ull,
            matrix_fragment_92, matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6600ull,
            matrix_accumulator_100, matrix_accumulator_101, matrix_accumulator_102,
            matrix_accumulator_103, memory_ranges, error_counts);

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_107 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(0u));

        packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                        select_half_operand<2, false, false>(0u));

        matrix_accumulator_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                          select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                          select_half_operand<2, false, false>(matrix_accumulator_107));

        matrix_accumulator_107 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                          select_half_operand<2, false, false>(packed_half_153));

        multiply_accumulate_half(
            matrix_accumulator_116, matrix_accumulator_117, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
            matrix_fragment_44, matrix_fragment_45, matrix_accumulator_116, matrix_accumulator_117);

        multiply_accumulate_half(
            matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_104,
            matrix_accumulator_105, matrix_accumulator_106, matrix_accumulator_107,
            matrix_fragment_46, matrix_fragment_47, matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_accumulator_48, matrix_accumulator_49,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_72, matrix_fragment_73, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_74, matrix_fragment_75, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_72, matrix_fragment_73, 0u, 0u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_84, matrix_fragment_85, matrix_fragment_60,
                                 matrix_fragment_61);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_74, matrix_fragment_75, 0u, 0u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_accumulator_104, matrix_accumulator_105,
                                 matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_accumulator_50, matrix_accumulator_51,
                                 matrix_accumulator_124, matrix_accumulator_125);

        matrix_accumulator_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_62,
                                 matrix_fragment_63);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_51),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_accumulator_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_43),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_48,
                                 matrix_accumulator_49);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_51),
            select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

        matrix_accumulator_51 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43),
            select_half_operand<2, false, false>(matrix_accumulator_50), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_56,
                                 matrix_fragment_57);

        matrix_accumulator_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

        matrix_accumulator_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_51), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_72,
                                 matrix_fragment_73);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_50), 0xc400c400u);

        matrix_accumulator_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_51),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_59 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

        matrix_accumulator_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_51),
            select_half_operand<2, false, false>(matrix_accumulator_50), 0x3b283b28u);

        matrix_accumulator_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_70, matrix_fragment_71, matrix_fragment_40,
                                 matrix_fragment_41);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

        matrix_accumulator_51 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_accumulator_51), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_48), 0x44004400u);

        matrix_fragment_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_accumulator_50));

        matrix_accumulator_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                select_half_operand<2, false, false>(matrix_accumulator_51));

        matrix_fragment_69 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_41 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_49), 0x44004400u);

        matrix_fragment_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0x44004400u);

        matrix_accumulator_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_50), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_43 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_41 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_51),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_41),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_51),
            select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

        matrix_accumulator_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_43),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                          select_half_operand<2, false, false>(matrix_accumulator_51));

        matrix_accumulator_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43),
            select_half_operand<2, false, false>(matrix_accumulator_50), 0x3b283b28u);

        matrix_fragment_74 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_75 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_74), 0xc400c400u);

        matrix_fragment_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73), 0x44004400u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_47 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_69),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_10 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

        matrix_fragment_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_75),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_47),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_74), 0x3b283b28u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_accumulator_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                select_half_operand<2, false, false>(matrix_accumulator_50));

        matrix_fragment_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_75),
            select_half_operand<2, false, false>(matrix_fragment_80), 0x3b283b28u);

        packed_half_147 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_10),
            select_half_operand<0, false, true>(packed_half_147), 0x37283728u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

        matrix_accumulator_48 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47),
            select_half_operand<2, false, false>(matrix_accumulator_48), 0x3b283b28u);

        matrix_accumulator_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_50),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                select_half_operand<2, false, false>(matrix_fragment_74));

        matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                select_half_operand<2, false, false>(matrix_fragment_80));

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        packed_half_147 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_10),
            select_half_operand<2, false, false>(packed_half_147), 0x3b283b28u);

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_accumulator_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                select_half_operand<2, false, false>(matrix_accumulator_48));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                          select_half_operand<2, false, false>(matrix_accumulator_49));

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                          select_half_operand<2, false, false>(matrix_fragment_58));

        packed_half_147 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                select_half_operand<2, false, false>(packed_half_147));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                          select_half_operand<2, false, false>(matrix_accumulator_48));

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_120,
                                 matrix_accumulator_121);

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_accumulator_49));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                          select_half_operand<2, false, false>(matrix_accumulator_48));

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_108,
                                 matrix_accumulator_109);

        uniform_condition_1 = uniform_condition_0;

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_18,
                                 matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_98, matrix_accumulator_99,
                                 matrix_accumulator_28, matrix_accumulator_29);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_102, matrix_accumulator_103,
                                 matrix_accumulator_30, matrix_accumulator_31);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_38,
                                 matrix_fragment_39);

        matrix_fragment_10 = 0x1u;

        uniform_condition_0 = false;
    } while (uniform_condition_1);
    thread_row_165 = threadIdx.y;

    uniform_index_word_5 = 0u;

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_4 = 0x400u;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_4 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = thread_row_165 * 0x800u;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 = packed_half_3 + uniform_index_word_4;

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_18 = address_word_159 * 0x10u + matrix_accumulator_16;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 = thread_row_165 * 0x300u;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                      select_half_operand<2, false, false>(matrix_fragment_4));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_18, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_accumulator_17));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_18 + 0x200u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    packed_half_0 = packed_half_0 & 0xfffffffbu;

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_18 + 0x400u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                        int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    packed_half_2 = (address_word_159 << 0x4u) + uniform_index_word_4;

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_18 + 0x600u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    condition_3 = (address_word_159 & 0x1u) != 0;

    condition_2 = (address_word_159 & 0x2u) != 0;

    condition_1 = (address_word_159 & 0x8u) != 0;

    condition_0 = (address_word_159 & 0x10u) != 0;

    condition_6 = int32_t(index_word_151) >= int32_t(index_word_157);

    if (condition_4) {
        packed_half_0 = 0x4u | packed_half_0;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x72a0ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x70a0ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x74a0ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x76a0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x78a0ull,
        matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x7aa0ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x8aa0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x92a0ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x90a0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x88a0ull,
        matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_50, matrix_accumulator_51,
        memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x8ca0ull,
        matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_122,
        matrix_accumulator_123, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x8ea0ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x200u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x400u, matrix_fragment_24,
                                        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0xc00u, matrix_fragment_64,
                                        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_28, matrix_accumulator_29, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_30, matrix_accumulator_31, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_16,
                             matrix_accumulator_17);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_18,
                             matrix_accumulator_19);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_56, matrix_fragment_57, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_14,
                             matrix_fragment_15);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x600u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_28, matrix_accumulator_29, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_30, matrix_accumulator_31, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_60,
                             matrix_fragment_61);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x800u, matrix_fragment_60,
                                        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_104,
                             matrix_accumulator_105);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_112,
                             matrix_accumulator_113);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_118,
                             matrix_accumulator_119);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0xa00u, matrix_fragment_68,
                                        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_accumulator_103),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    matrix_fragment_9 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_78 = result_word;
    }

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(thread_row_165)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0xe00u, matrix_fragment_72,
                                        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    packed_half_158 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0xe0a0ull, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_8 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5432u);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x4100410u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_9 & 65535u) * 65537u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x4100410u);

    matrix_fragment_15 = half_to_float_bits((matrix_fragment_9 >> 16) * 65537u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_83 = half_to_float_bits((matrix_fragment_14 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_80 = result_word;
    }

    packed_half_138 = half_to_float_bits((matrix_fragment_14 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_4, matrix_fragment_5, 0u, 0u);

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    matrix_accumulator_141 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                       select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_11 = rsqrt_bits(matrix_fragment_15);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_132,
                             matrix_accumulator_133);

    matrix_fragment_83 = rsqrt_bits(matrix_fragment_83);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_141), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_134,
                             matrix_accumulator_135);

    packed_half_138 = rsqrt_bits(packed_half_138);

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    packed_half_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_128,
                             matrix_fragment_129);

    packed_half_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_130,
                             matrix_accumulator_131);

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(packed_half_3));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    packed_half_139 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(packed_half_139));

    matrix_fragment_10 = pack_half(matrix_fragment_11, matrix_fragment_10);

    matrix_fragment_11 = pack_half(packed_half_138, matrix_fragment_83);

    matrix_fragment_83 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                     select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_132));

    packed_half_138 = permute_signed(matrix_fragment_83, matrix_fragment_83, 0x5432u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_6, matrix_fragment_7, 0u, 0u);

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_128),
                                      select_half_operand<2, false, false>(matrix_fragment_128),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_141 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                       select_half_operand<2, false, false>(matrix_fragment_82));

    packed_half_142 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                     select_half_operand<2, false, false>(packed_half_138));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_4,
                             matrix_fragment_5);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_139), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_accumulator_140 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_129),
                                      select_half_operand<2, false, false>(matrix_fragment_129),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_141), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_144 = result_word;
    }

    packed_half_142 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_142), 0x4100410u);

    packed_half_138 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                     select_half_operand<2, false, false>(matrix_fragment_5));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_140), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_147 = result_word;
    }

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_14, matrix_fragment_15));

    __syncthreads();

    packed_half_143 = half_to_float_bits((packed_half_142 & 65535u) * 65537u);

    packed_half_146 = half_to_float_bits((packed_half_142 >> 16) * 65537u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_138), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_145 = result_word;
    }

    packed_half_143 = rsqrt_bits(packed_half_143);

    packed_half_3 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5432u);

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                     select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_146 = rsqrt_bits(packed_half_146);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xa2a0ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                   select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_139), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_142 = result_word;
    }

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_141 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                       select_half_operand<2, false, false>(packed_half_144));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xa0a0ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x4100410u);

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                     select_half_operand<2, false, false>(packed_half_147));

    packed_half_145 = add_half_pairs(select_half_operand<2, false, false>(packed_half_138),
                                     select_half_operand<2, false, false>(packed_half_145));

    matrix_accumulator_140 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    packed_half_147 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_fragment_12 = permute_signed(matrix_accumulator_141, matrix_accumulator_141, 0x5432u);

    packed_half_138 = pack_half(packed_half_146, packed_half_143);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    matrix_fragment_88 = rsqrt_bits(matrix_accumulator_140);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_144), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                     select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(packed_half_138));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_145), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_96 = result_word;
    }

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_fragment_11));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xa4a0ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    packed_half_143 = rsqrt_bits(packed_half_147);

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                     select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(packed_half_138));

    matrix_accumulator_98 = permute_signed(packed_half_139, packed_half_139, 0x5432u);

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                     select_half_operand<2, false, false>(matrix_accumulator_98));

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                     select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_146), 0x4100410u);

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    packed_half_139 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_139), 0x4100410u);

    packed_half_146 = pack_half(packed_half_143, matrix_fragment_88);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_136,
                             matrix_accumulator_137);

    packed_half_143 = permute_signed(packed_half_144, packed_half_144, 0x5432u);

    packed_half_154 = half_to_float_bits((matrix_fragment_90 & 65535u) * 65537u);

    packed_half_153 = half_to_float_bits((packed_half_139 & 65535u) * 65537u);

    matrix_accumulator_98 = half_to_float_bits((matrix_fragment_90 >> 16) * 65537u);

    matrix_fragment_92 = half_to_float_bits((packed_half_139 >> 16) * 65537u);

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                     select_half_operand<2, false, false>(packed_half_143));

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_94, matrix_fragment_95, matrix_accumulator_140,
                             matrix_accumulator_141);

    packed_half_154 = rsqrt_bits(packed_half_154);

    packed_half_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(packed_half_146));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xa6a0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_95 = permute_signed(matrix_accumulator_96, matrix_accumulator_96, 0x5432u);

    matrix_fragment_93 = rsqrt_bits(matrix_accumulator_98);

    matrix_accumulator_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_144), 0x4100410u);

    packed_half_143 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                     select_half_operand<2, false, false>(matrix_fragment_95));

    packed_half_153 = rsqrt_bits(packed_half_153);

    packed_half_143 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_143), 0x4100410u);

    matrix_fragment_92 = rsqrt_bits(matrix_fragment_92);

    packed_half_149 = half_to_float_bits((matrix_accumulator_97 & 65535u) * 65537u);

    packed_half_148 = half_to_float_bits((matrix_accumulator_97 >> 16) * 65537u);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_accumulator_104));

    packed_half_152 = half_to_float_bits((packed_half_143 & 65535u) * 65537u);

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    packed_half_150 = half_to_float_bits((packed_half_143 >> 16) * 65537u);

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_fragment_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_95 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_95));

    packed_half_148 = rsqrt_bits(packed_half_148);

    packed_half_147 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                     select_half_operand<2, false, false>(matrix_fragment_94));

    packed_half_154 = pack_half(matrix_fragment_93, packed_half_154);

    packed_half_153 = pack_half(matrix_fragment_92, packed_half_153);

    packed_half_144 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(packed_half_146));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xa8a0ull, matrix_fragment_92,
        matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges, error_counts);

    packed_half_152 = rsqrt_bits(packed_half_152);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xaaa0ull, matrix_accumulator_96,
        matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, memory_ranges,
        error_counts);

    packed_half_150 = rsqrt_bits(packed_half_150);

    packed_half_143 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_108 = pack_half(packed_half_148, packed_half_149);

    packed_half_148 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(packed_half_138));

    packed_half_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(packed_half_146));

    packed_half_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(packed_half_138));

    packed_half_146 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(packed_half_146));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xaca0ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    packed_half_145 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_109 = pack_half(packed_half_150, packed_half_152);

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_154));

    packed_half_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_153));

    packed_half_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_128),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    packed_half_153 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_129),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xaea0ull, matrix_accumulator_108,
        matrix_accumulator_109, matrix_accumulator_110, matrix_accumulator_111, memory_ranges,
        error_counts);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_147), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_130 = result_word;
    }

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_129 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_128));

    matrix_fragment_11 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    packed_half_138 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_138),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_129),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(packed_half_138));

    matrix_fragment_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_138 = result_word;
    }

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_accumulator_140));

    packed_half_142 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_129 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_129));

    matrix_accumulator_131 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_128 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_139),
                                        select_half_operand<2, false, false>(packed_half_142));

    packed_half_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                     select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                       select_half_operand<2, false, false>(matrix_fragment_129));

    matrix_fragment_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_accumulator_141));

    packed_half_142 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(packed_half_142));

    matrix_accumulator_131 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_fragment_129));

    packed_half_147 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                    select_half_operand<2, false, false>(0u));

    packed_half_142 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                     select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_129 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                        select_half_operand<2, false, false>(packed_half_147));

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                     select_half_operand<2, false, false>(packed_half_138));

    matrix_fragment_11 = packed_half_142;

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_139), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_144 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_130), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_131 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    packed_half_138 = pack_half(0u, packed_half_158);

    matrix_fragment_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                            select_half_operand<0, false, false>(packed_half_138));

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                     select_half_operand<2, false, false>(packed_half_144));

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                     select_half_operand<2, false, false>(matrix_accumulator_131));

    packed_half_142 = add_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                     select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                             select_half_operand<0, false, false>(packed_half_138));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_144), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_3 = result_word;
    }

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_142), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_147 = result_word;
    }

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                            select_half_operand<0, false, false>(packed_half_138));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_146), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_4 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_6 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                   select_half_operand<2, false, false>(packed_half_3));

    packed_half_144 = permute_signed(packed_half_3, packed_half_3, 0x5432u);

    packed_half_142 = add_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                     select_half_operand<2, false, false>(packed_half_147));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                   select_half_operand<2, false, false>(packed_half_144));

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    packed_half_158 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                           select_half_operand<2, false, false>(0u));

    packed_half_144 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x4100410u);

    packed_half_3 = permute_signed(packed_half_142, packed_half_142, 0x5432u);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_125 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 = add_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                            select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(packed_half_142, packed_half_143, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_80, matrix_fragment_81);

    packed_half_3 = half_to_float_bits((packed_half_144 & 65535u) * 65537u);

    matrix_fragment_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x4100410u);

    matrix_accumulator_134 = half_to_float_bits((packed_half_144 >> 16) * 65537u);

    packed_half_3 = rsqrt_bits(packed_half_3);

    packed_half_144 = permute_signed(packed_half_139, packed_half_139, 0x5432u);

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_134);

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 = permute_signed(packed_half_146, packed_half_146, 0x5432u);

    packed_half_139 = add_half_pairs(select_half_operand<2, false, false>(packed_half_139),
                                     select_half_operand<2, false, false>(packed_half_144));

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_80 = half_to_float_bits((matrix_fragment_81 & 65535u) * 65537u);

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_146 = add_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                     select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_81 = half_to_float_bits((matrix_fragment_81 >> 16) * 65537u);

    packed_half_139 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_139), 0x4100410u);

    packed_half_146 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_146), 0x4100410u);

    matrix_fragment_80 = rsqrt_bits(matrix_fragment_80);

    packed_half_145 = pack_half(matrix_accumulator_134, packed_half_3);

    packed_half_144 = half_to_float_bits((packed_half_139 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_81 = rsqrt_bits(matrix_fragment_81);

    matrix_fragment_84 = half_to_float_bits((packed_half_139 >> 16) * 65537u);

    packed_half_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_fragment_85 = half_to_float_bits((packed_half_146 & 65535u) * 65537u);

    matrix_accumulator_122 = half_to_float_bits((packed_half_146 >> 16) * 65537u);

    packed_half_3 = rsqrt_bits(packed_half_144);

    matrix_fragment_84 = rsqrt_bits(matrix_fragment_84);

    matrix_fragment_85 = rsqrt_bits(matrix_fragment_85);

    matrix_accumulator_122 = rsqrt_bits(matrix_accumulator_122);

    packed_half_144 = pack_half(matrix_fragment_81, matrix_fragment_80);

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(packed_half_145));

    packed_half_139 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_139),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(packed_half_144));

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(packed_half_144));

    packed_half_3 = pack_half(matrix_fragment_84, packed_half_3);

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_145));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                                       select_half_operand<2, false, false>(packed_half_139));

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(packed_half_144));

    packed_half_149 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_144));

    packed_half_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_122 = pack_half(matrix_accumulator_122, matrix_fragment_85);

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_124 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                           select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_132 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                           select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                                       select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(packed_half_138));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_accumulator_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<0, false, false>(packed_half_138));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<0, false, false>(packed_half_138));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_128, matrix_fragment_129, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_104),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                      select_half_operand<2, false, false>(matrix_accumulator_106));

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_128, matrix_fragment_129, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_92,
                             matrix_fragment_93);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_98,
                             matrix_accumulator_99);

    packed_half_3 = 0x29c0u;

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_142 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_142),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3c203c20u);

    matrix_fragment_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_94),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_accumulator_108,
                             matrix_accumulator_109);

    packed_half_142 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_142), 0x3c203c20u);

    matrix_accumulator_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3c203c20u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_95),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_98),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_accumulator_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3e473e47u);

    packed_half_142 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_142), 0x3e473e47u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3e473e47u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_94 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3c203c20u);

    matrix_accumulator_134 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3c203c20u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3c203c20u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3c203c20u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    packed_half_143 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_143),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_96 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3c203c20u);

    matrix_accumulator_98 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_98), 0x3c203c20u);

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_fragment_87 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_94 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x3e473e47u);

    matrix_accumulator_97 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3c203c20u);

    matrix_accumulator_99 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99), 0x3c203c20u);

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3e473e47u);

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3e473e47u);

    matrix_fragment_78 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_83 = (matrix_fragment_24 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = (packed_half_142 << 0x5u) + 0x7ff88000u;

    matrix_fragment_79 = (matrix_fragment_25 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_95 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x3e473e47u);

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_accumulator_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3e473e47u);

    matrix_accumulator_98 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_98), 0x3e473e47u);

    matrix_accumulator_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    packed_half_143 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_143), 0x3c203c20u);

    matrix_accumulator_97 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3e473e47u);

    matrix_accumulator_99 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99), 0x3e473e47u);

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_8 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_105 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                       select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_10 = (matrix_fragment_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = (matrix_fragment_26 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_94 = (matrix_fragment_95 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_11 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_93 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 = (matrix_accumulator_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    packed_half_143 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_143), 0x3e473e47u);

    matrix_fragment_9 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_accumulator_135 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3c203c20u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_88 = (matrix_accumulator_97 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_92 = (matrix_accumulator_99 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_105 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                       select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3e473e47u);

    matrix_accumulator_122 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_77 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_82 = (packed_half_143 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_86 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_123 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_accumulator_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_102 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3c203c20u);

    matrix_accumulator_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_accumulator_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_accumulator_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3c203c20u);

    matrix_accumulator_121 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_accumulator_122 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_108 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108), 0x3c203c20u);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                       select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_accumulator_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                       select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_102 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3e473e47u);

    matrix_accumulator_123 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                       select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_110 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3c203c20u);

    matrix_accumulator_121 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                       select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_25 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    matrix_fragment_27 = (matrix_fragment_27 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_accumulator_97 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_accumulator_103 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3e473e47u);

    matrix_accumulator_109 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109), 0x3c203c20u);

    matrix_accumulator_111 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_accumulator_104 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108), 0x3e473e47u);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_accumulator_110 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3e473e47u);

    matrix_accumulator_108 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_120 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_91 = (matrix_accumulator_102 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_109 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109), 0x3e473e47u);

    matrix_accumulator_111 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3e473e47u);

    matrix_accumulator_107 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_89 = (matrix_accumulator_97 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_120 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_accumulator_120));

    matrix_accumulator_97 = (matrix_accumulator_104 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_96 = (matrix_accumulator_103 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_105 = (matrix_accumulator_110 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_102 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_104 = (matrix_accumulator_109 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_106 = (matrix_accumulator_111 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_102 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                       select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_103 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_123 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                       select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_accumulator_109 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                       select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_accumulator_102 = address_word_159 * 0x4u;

    matrix_fragment_32 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_110 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                       select_half_operand<2, false, false>(matrix_accumulator_110));

    matrix_accumulator_122 = matrix_accumulator_102 & 0x1cu;

    matrix_accumulator_111 = condition_3 ? matrix_fragment_32 : matrix_accumulator_121;

    matrix_accumulator_120 = condition_3 ? matrix_accumulator_110 : matrix_accumulator_109;

    matrix_accumulator_121 = condition_3 ? matrix_accumulator_121 : matrix_fragment_32;

    matrix_accumulator_110 = condition_3 ? matrix_accumulator_109 : matrix_accumulator_110;

    matrix_fragment_32 = uint32_t(int64_t(int32_t(address_word_159)) * (1ll << 0x1du) >> 32) +
                         matrix_accumulator_122;

    matrix_accumulator_123 = !condition_2 ? matrix_accumulator_111 : matrix_accumulator_120;

    matrix_accumulator_135 = !condition_2 ? matrix_accumulator_120 : matrix_accumulator_111;

    matrix_accumulator_120 = !condition_2 ? matrix_accumulator_121 : matrix_accumulator_110;

    matrix_accumulator_111 = 0x1u ^ matrix_fragment_32;

    matrix_accumulator_122 = !condition_2 ? matrix_accumulator_110 : matrix_accumulator_121;

    matrix_accumulator_110 = 0x2u ^ matrix_fragment_32;

    matrix_accumulator_109 = 0x3u ^ matrix_fragment_32;

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_28, matrix_accumulator_29, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_123), "r"(matrix_fragment_32), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_123 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_120), "r"(matrix_accumulator_111), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_120 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_135), "r"(matrix_accumulator_110), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_135 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_122), "r"(matrix_accumulator_109), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_122 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_28, matrix_accumulator_29, 0u, 0u);

    matrix_accumulator_121 = !condition_1 ? matrix_accumulator_123 : matrix_accumulator_120;

    matrix_accumulator_134 = !condition_1 ? matrix_accumulator_120 : matrix_accumulator_123;

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_130,
                             matrix_accumulator_131);

    matrix_accumulator_136 = !condition_1 ? matrix_accumulator_135 : matrix_accumulator_122;

    matrix_accumulator_123 = !condition_1 ? matrix_accumulator_122 : matrix_accumulator_135;

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_28,
                             matrix_accumulator_29);

    matrix_fragment_52 = !condition_0 ? matrix_accumulator_121 : matrix_accumulator_136;

    matrix_fragment_53 = !condition_0 ? matrix_accumulator_134 : matrix_accumulator_123;

    matrix_accumulator_121 = !condition_0 ? matrix_accumulator_136 : matrix_accumulator_121;

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_30, matrix_accumulator_31, 0u, 0u);

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_123 = !condition_0 ? matrix_accumulator_123 : matrix_accumulator_134;

    matrix_fragment_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_121));

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_30, matrix_accumulator_31, 0u, 0u);

    matrix_fragment_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_52 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_52),
                                        select_half_operand<1, false, false>(matrix_fragment_52));

    matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_52, 0x5410u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_53 = half_to_float_bits((matrix_fragment_52 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_30,
                             matrix_accumulator_31);

    matrix_fragment_53 = reciprocal_bits(matrix_fragment_53);

    matrix_fragment_54 = half_to_float_bits((matrix_fragment_52 >> 16) * 65537u);

    matrix_fragment_55 = uint32_t(int32_t(index_word_164) >> 0x2u);

    matrix_fragment_52 = matrix_fragment_55 + 0x8u;

    matrix_fragment_54 = reciprocal_bits(matrix_fragment_54);

    matrix_accumulator_120 = uint32_t(int32_t(matrix_fragment_52) >> 0x1fu);

    matrix_accumulator_122 = uint32_t(int32_t(index_word_164) >> 0x1fu);

    matrix_accumulator_121 =
        uint32_t(combine_words(matrix_accumulator_120, 0u) << 0x5u >> 32) + matrix_fragment_52;

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_accumulator_122 =
        uint32_t(combine_words(matrix_accumulator_122, 0u) << 0x5u >> 32) + matrix_fragment_55;

    matrix_accumulator_121 = matrix_accumulator_121 & 0xffffffe0u;

    matrix_accumulator_122 = matrix_accumulator_122 & 0xffffffe0u;

    matrix_accumulator_123 = matrix_fragment_52 + (0u - matrix_accumulator_121);

    matrix_accumulator_134 = pack_half(matrix_fragment_54, matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_53 = matrix_fragment_55 + (0u - matrix_accumulator_122);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(matrix_accumulator_123), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_123 = result_word;
    }

    matrix_accumulator_115 = transpose_half_tile(matrix_accumulator_115);

    matrix_accumulator_19 = transpose_half_tile(matrix_accumulator_19);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(matrix_fragment_53), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_121 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_98,
                             matrix_accumulator_99);

    matrix_accumulator_18 = transpose_half_tile(matrix_accumulator_18);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_45 = transpose_half_tile(matrix_accumulator_114);

    matrix_accumulator_122 =
        permute_signed(matrix_accumulator_123, matrix_accumulator_123, 0x5410u);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_121 =
        permute_signed(matrix_accumulator_121, matrix_accumulator_121, 0x5410u);

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_fragment_47 = transpose_half_tile(matrix_accumulator_17);

    matrix_fragment_46 = transpose_half_tile(matrix_accumulator_16);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_76 = uint32_t(int64_t(int32_t(index_word_164)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                          select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_77 = uint32_t(int32_t(matrix_fragment_76) >> 0x1fu);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_82 = uint32_t(int64_t(int32_t(index_word_164)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_77 =
        uint32_t(combine_words(matrix_fragment_77, 0u) << 0x5u >> 32) + matrix_fragment_76;

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_83 = uint32_t(int32_t(matrix_fragment_82) >> 0x1fu);

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_77 = matrix_fragment_77 & 0xffffffe0u;

    matrix_fragment_78 = transpose_half_tile(matrix_accumulator_20);

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_accumulator_117 = transpose_half_tile(matrix_accumulator_117);

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                          select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_46 =
        uint32_t(combine_words(matrix_fragment_83, 0u) << 0x5u >> 32) + matrix_fragment_82;

    matrix_fragment_83 = matrix_fragment_76 + (0u - matrix_fragment_77);

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = transpose_half_tile(matrix_accumulator_116);

    matrix_fragment_87 = matrix_fragment_46 & 0xffffffe0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(matrix_fragment_83), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_47 = transpose_half_tile(matrix_accumulator_23);

    matrix_accumulator_113 = transpose_half_tile(matrix_accumulator_113);

    matrix_fragment_79 = transpose_half_tile(matrix_accumulator_21);

    matrix_accumulator_119 = transpose_half_tile(matrix_accumulator_119);

    matrix_fragment_87 = matrix_fragment_82 + (0u - matrix_fragment_87);

    matrix_fragment_46 = transpose_half_tile(matrix_accumulator_22);

    matrix_accumulator_112 = transpose_half_tile(matrix_accumulator_112);

    matrix_accumulator_118 = transpose_half_tile(matrix_accumulator_118);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(matrix_fragment_87), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                      select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_116 = permute_signed(matrix_fragment_83, matrix_fragment_83, 0x5410u);

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                      select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_47 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                      select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = transpose_half_tile(matrix_fragment_7);

    matrix_accumulator_117 = permute_signed(matrix_fragment_87, matrix_fragment_87, 0x5410u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_6 = transpose_half_tile(matrix_fragment_6);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_87 = transpose_half_tile(matrix_accumulator_30);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_131 = transpose_half_tile(matrix_accumulator_131);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_29 = transpose_half_tile(matrix_accumulator_29);

    matrix_accumulator_99 = transpose_half_tile(matrix_accumulator_99);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_accumulator_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = transpose_half_tile(matrix_accumulator_31);

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_101 = transpose_half_tile(matrix_accumulator_101);

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_103 = transpose_half_tile(matrix_accumulator_103);

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_35);

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 = transpose_half_tile(matrix_accumulator_130);

    matrix_accumulator_28 = transpose_half_tile(matrix_accumulator_28);

    matrix_fragment_92 = transpose_half_tile(matrix_accumulator_98);

    matrix_fragment_93 = transpose_half_tile(matrix_accumulator_100);

    matrix_fragment_94 = transpose_half_tile(matrix_accumulator_102);

    matrix_fragment_34 = transpose_half_tile(matrix_fragment_34);

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_fragment_90));

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                      select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(matrix_accumulator_18));

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_94 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                      select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                      select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_28),
                                       select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_accumulator_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_35 = thread_row_165 * 0x20u;

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_94, matrix_fragment_95, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_105),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                      select_half_operand<2, false, false>(matrix_accumulator_106));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_34 = uint32_t(uint32_t(matrix_fragment_35) >> 0x1u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_94, matrix_fragment_95, matrix_fragment_6,
                             matrix_fragment_7);

    matrix_fragment_6 = index_word_164 & 0x7ffffffcu;

    matrix_fragment_6 = 0u - matrix_fragment_6 + address_word_159;

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_112,
                             matrix_accumulator_113);

    matrix_fragment_7 = matrix_fragment_34 + 0x4u;

    matrix_fragment_6 = matrix_fragment_6 * 0x2u + matrix_fragment_35;

    matrix_fragment_33 = matrix_fragment_34 + 0x8u;

    matrix_fragment_34 = matrix_fragment_34 + 0xcu;

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_30,
                             matrix_accumulator_31);

    matrix_accumulator_29 = matrix_fragment_33 + address_word_159 + (0u - index_word_163);

    matrix_fragment_79 = matrix_fragment_34 + address_word_159 + (0u - index_word_163);

    matrix_fragment_89 = thread_row_165 * 0x100u;

    matrix_fragment_11 = matrix_fragment_7 + address_word_159 + (0u - index_word_163);

    matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_6) >> 0x1u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_6 = (thread_row_165 << 0x9u) + uniform_index_word_4;

    matrix_accumulator_99 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                        uint64_t(uint32_t(matrix_accumulator_29)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_101 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_78, matrix_fragment_79,
                        uint64_t(uint32_t(matrix_fragment_79)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_98 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_28, matrix_accumulator_29) + 0xf0b0ull, memory_ranges,
        error_counts);

    split_address_words(address_word_160, address_word_161,
                        uint64_t(uint32_t(matrix_fragment_89)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_78, matrix_fragment_79) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = address_word_159 * 0x10u + matrix_fragment_6;

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                          select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_6, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_6, matrix_accumulator_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_6 + 0x400u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_6 + 0x400u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_82, matrix_fragment_83,
                        int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_160, address_word_161));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe0b0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe2b0ull, matrix_accumulator_28,
        matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe8b0ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xeab0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_56, matrix_fragment_57, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_9);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_50, matrix_accumulator_51, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_104,
                             matrix_accumulator_105);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_accumulator_50, matrix_accumulator_51, matrix_accumulator_112,
                             matrix_accumulator_113);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_99));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_accumulator_49 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_accumulator_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                            select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_40,
                             matrix_fragment_41);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x400u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_9 = unpack_fp8_bits(matrix_accumulator_17);

    matrix_accumulator_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_fragment_8 = unpack_fp8_bits(matrix_accumulator_16);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_101));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_16,
                             matrix_accumulator_17);

    matrix_fragment_74 = blockIdx.y;

    matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_18);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_accumulator_18 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    matrix_accumulator_19 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x200u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_50,
                             matrix_accumulator_51);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_2 + 0x600u, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_28, matrix_accumulator_29, matrix_accumulator_16,
                             matrix_accumulator_17);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_30, matrix_accumulator_31, matrix_accumulator_18,
                             matrix_accumulator_19);

    matrix_fragment_43 = parameters[parameter_geometry_or_control_11];

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_74 = matrix_fragment_74 * 0x8u + matrix_fragment_43;

    condition_6 = int32_t(matrix_fragment_74) > int32_t(0u - 0x4u) && !condition_6;

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_24,
                             matrix_fragment_25);

    condition_4 = condition_6 & condition_5;

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                      select_half_operand<2, false, false>(matrix_fragment_62),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_0 = packed_half_0 & 0xfffffffdu;

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_73));

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_accumulator_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                           select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    if (condition_4) {
        packed_half_0 = 0x2u | packed_half_0;
    }

    matrix_fragment_40 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                      select_half_operand<2, false, false>(matrix_fragment_71),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_accumulator_96));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_28,
                             matrix_accumulator_29);

    matrix_fragment_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(matrix_fragment_63),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_27 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_accumulator_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_accumulator_28));

    condition_4 = (packed_half_0 & 0x4u) != 0;

    matrix_accumulator_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                           select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_40 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                       select_half_operand<2, false, false>(matrix_accumulator_28));

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_90, matrix_fragment_91, matrix_accumulator_30,
                             matrix_accumulator_31);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_99), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_28 = result_word;
    }

    condition_6 = condition_6 & condition_4;

    matrix_fragment_26 = parameters[parameter_output_low_2];
    matrix_fragment_27 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_98), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_29 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_41), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_36 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    packed_half_0 = packed_half_0 & 0xfffffffeu;

    if (condition_6) {
        packed_half_0 = 0x1u | packed_half_0;
    }

    condition_6 = (packed_half_0 & 0x2u) != 0;

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_50,
                             matrix_accumulator_51);

    matrix_fragment_37 = index_word_151 * index_word_155 + index_word_156;

    matrix_accumulator_99 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                       select_half_operand<2, false, false>(matrix_accumulator_28));

    packed_half_2 = thread_row_165 * 0x80u;

    matrix_accumulator_98 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                       select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_37 = matrix_fragment_37 * 0x100u + packed_half_2;

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_90, matrix_fragment_91, matrix_accumulator_48,
                             matrix_accumulator_49);

    matrix_accumulator_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                          select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_99), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_90 = result_word;
    }

    split_address_words(matrix_fragment_26, matrix_fragment_27,
                        int64_t(int32_t(matrix_fragment_37)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_26, matrix_fragment_27));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_98), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_91 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_76 = result_word;
    }
    if (condition_6) {
        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_51),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_accumulator_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                          select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_26, matrix_fragment_27));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_50),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_9 = matrix_accumulator_18;

        matrix_fragment_11 = matrix_accumulator_51;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_16, matrix_accumulator_17), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    condition_6 = (packed_half_0 & 0x1u) != 0;

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                      select_half_operand<2, false, false>(0u));
    if (condition_6) {
        matrix_fragment_25 = permute_signed(matrix_fragment_25, matrix_fragment_8, 0x5410u);

        split_address_words(matrix_fragment_26, matrix_fragment_27,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_26, matrix_fragment_27));

        packed_half_0 = permute_signed(matrix_fragment_77, packed_half_0, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_fragment_24, matrix_fragment_11, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_accumulator_28, matrix_fragment_9, 0x5410u);

        matrix_fragment_9 = packed_half_0;

        matrix_fragment_11 = matrix_fragment_25;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_26, matrix_fragment_27) + 0x400ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xb6a0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xb0a0ull, matrix_accumulator_48,
        matrix_accumulator_49, matrix_accumulator_50, matrix_accumulator_51, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xb2a0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xb4a0ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xbaa0ull, matrix_accumulator_28,
        matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xb8a0ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xbca0ull, matrix_accumulator_16,
        matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xbea0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                        select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_14 = permute_signed(matrix_fragment_91, matrix_fragment_91, 0x5432u);

    matrix_fragment_15 = permute_signed(matrix_fragment_90, matrix_fragment_90, 0x5432u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x4100410u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x4100410u);

    matrix_fragment_14 = half_to_float_bits((matrix_fragment_91 & 65535u) * 65537u);

    matrix_fragment_91 = half_to_float_bits((matrix_fragment_91 >> 16) * 65537u);

    packed_half_0 = half_to_float_bits((matrix_fragment_90 & 65535u) * 65537u);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_90 >> 16) * 65537u);

    matrix_fragment_14 = rsqrt_bits(matrix_fragment_14);

    matrix_fragment_91 = rsqrt_bits(matrix_fragment_91);

    packed_half_0 = rsqrt_bits(packed_half_0);

    matrix_fragment_15 = rsqrt_bits(matrix_fragment_90);

    matrix_fragment_77 = pack_half(matrix_fragment_91, matrix_fragment_14);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_15 = pack_half(matrix_fragment_15, packed_half_0);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(packed_half_138));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_60 = permute_signed(matrix_fragment_64, matrix_fragment_15, 0x5410u);

    matrix_fragment_63 = permute_signed(matrix_fragment_63, matrix_fragment_14, 0x5410u);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_15 = permute_signed(matrix_fragment_88, matrix_fragment_88, 0x5432u);

    matrix_fragment_14 = permute_signed(matrix_fragment_75, matrix_fragment_75, 0x5432u);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x4100410u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x4100410u);

    matrix_fragment_61 = permute_signed(matrix_fragment_65, packed_half_0, 0x5410u);

    packed_half_0 = half_to_float_bits((matrix_fragment_88 & 65535u) * 65537u);

    matrix_fragment_64 = half_to_float_bits((matrix_fragment_75 & 65535u) * 65537u);

    matrix_fragment_75 = half_to_float_bits((matrix_fragment_75 >> 16) * 65537u);

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_88 >> 16) * 65537u);

    packed_half_0 = rsqrt_bits(packed_half_0);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_65 = rsqrt_bits(matrix_fragment_88);

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_64 = rsqrt_bits(matrix_fragment_64);

    matrix_fragment_75 = rsqrt_bits(matrix_fragment_75);

    matrix_fragment_62 = permute_signed(matrix_fragment_62, matrix_fragment_77, 0x5410u);

    packed_half_0 = pack_half(matrix_fragment_65, packed_half_0);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(packed_half_138));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = pack_half(matrix_fragment_75, matrix_fragment_64);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(packed_half_138));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<0, false, false>(packed_half_138));

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_48,
                             matrix_accumulator_49);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_50, matrix_accumulator_51, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_128, matrix_fragment_129, matrix_accumulator_50,
                             matrix_accumulator_51);

    matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

    matrix_fragment_9 = permute_signed(matrix_fragment_67, packed_half_0, 0x5410u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_38,
                             matrix_fragment_39);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = permute_signed(matrix_fragment_10, matrix_fragment_11, 0x5410u);

    matrix_fragment_11 = permute_signed(matrix_accumulator_97, matrix_fragment_62, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_28, matrix_accumulator_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_28,
                             matrix_accumulator_29);

    multiply_accumulate_half(matrix_accumulator_30, matrix_accumulator_31, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_30,
                             matrix_accumulator_31);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_128, matrix_fragment_129, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_accumulator_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_51),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_49),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_16,
                             matrix_accumulator_17);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_51 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_51), 0x3c203c20u);

    matrix_accumulator_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_49 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_49), 0x3c203c20u);

    matrix_accumulator_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_48),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3c203c20u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3c203c20u);

    matrix_accumulator_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_30),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_18,
                             matrix_accumulator_19);

    matrix_accumulator_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_50),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3c203c20u);

    matrix_accumulator_51 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_51), 0x3e473e47u);

    matrix_accumulator_28 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28), 0x3c203c20u);

    matrix_accumulator_48 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_48), 0x3c203c20u);

    matrix_accumulator_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_49), 0x3e473e47u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3c203c20u);

    packed_half_0 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3e473e47u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_80, matrix_fragment_81, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_accumulator_30 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_30), 0x3c203c20u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x3e473e47u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_50 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_50), 0x3c203c20u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_13 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_28), 0x3e473e47u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_48 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_48), 0x3e473e47u);

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3e473e47u);

    matrix_accumulator_30 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_30), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_27 = (matrix_accumulator_51 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 = (matrix_accumulator_49 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_16),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_28 = (packed_half_0 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_51 = (matrix_fragment_5 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_50 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_50), 0x3e473e47u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3c203c20u);

    matrix_accumulator_49 = (matrix_fragment_4 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3c203c20u);

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3c203c20u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_4 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_31),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_12 = (matrix_accumulator_48 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_48 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                       select_half_operand<2, false, false>(matrix_accumulator_51));

    matrix_fragment_13 = (matrix_accumulator_30 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_30 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3e473e47u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x3e473e47u);

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x3e473e47u);

    matrix_accumulator_29 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29), 0x3c203c20u);

    matrix_accumulator_16 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_16), 0x3c203c20u);

    matrix_fragment_25 = (matrix_accumulator_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = (matrix_fragment_26 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_48 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_31 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_31), 0x3c203c20u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_8 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_29), 0x3e473e47u);

    matrix_accumulator_30 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_16), 0x3e473e47u);

    packed_half_0 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_5 = (matrix_fragment_42 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                       select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_accumulator_16 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_29 = (matrix_fragment_38 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3c203c20u);

    matrix_fragment_9 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_31), 0x3e473e47u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3c203c20u);

    matrix_accumulator_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_18),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_38 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x3e473e47u);

    matrix_fragment_36 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3e473e47u);

    matrix_fragment_41 = (matrix_fragment_9 << 0x5u) + 0x7ff88000u;

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_accumulator_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_18 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_18), 0x3c203c20u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_19),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_31 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_accumulator_50 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_39 = (matrix_fragment_43 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_15),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_10 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_18), 0x3e473e47u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_17 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17), 0x3c203c20u);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3c203c20u);

    matrix_fragment_42 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                       select_half_operand<2, false, false>(matrix_accumulator_50));

    matrix_accumulator_18 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3c203c20u);

    matrix_accumulator_50 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    matrix_accumulator_30 = (matrix_accumulator_30 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 = (matrix_fragment_10 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_17 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17), 0x3e473e47u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3e473e47u);

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x3c203c20u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_fragment_63 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                       select_half_operand<2, false, false>(matrix_accumulator_50));

    matrix_accumulator_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3e473e47u);

    matrix_fragment_10 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                       select_half_operand<2, false, false>(matrix_accumulator_31));

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x3e473e47u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_15 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    packed_half_3 = (matrix_accumulator_17 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_19 = (matrix_accumulator_19 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_50 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_48 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_17 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = (matrix_fragment_43 << 0x5u) + 0x7ff88000u;

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_accumulator_50));

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_9 = condition_3 ? matrix_fragment_11 : matrix_fragment_8;

    matrix_fragment_8 = condition_3 ? matrix_fragment_8 : matrix_fragment_11;

    matrix_fragment_10 = condition_3 ? matrix_fragment_57 : matrix_fragment_56;

    matrix_fragment_57 = condition_3 ? matrix_fragment_56 : matrix_fragment_57;

    matrix_fragment_11 = !condition_2 ? matrix_fragment_9 : matrix_fragment_10;

    matrix_fragment_56 = !condition_2 ? matrix_fragment_8 : matrix_fragment_57;

    matrix_fragment_9 = !condition_2 ? matrix_fragment_10 : matrix_fragment_9;

    matrix_fragment_8 = !condition_2 ? matrix_fragment_57 : matrix_fragment_8;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_56), "r"(matrix_accumulator_111), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_110), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_accumulator_109), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    matrix_fragment_10 = !condition_1 ? matrix_fragment_11 : matrix_fragment_56;

    matrix_fragment_57 = !condition_1 ? matrix_fragment_56 : matrix_fragment_11;

    matrix_fragment_59 = !condition_1 ? matrix_fragment_9 : matrix_fragment_8;

    matrix_fragment_58 = !condition_1 ? matrix_fragment_8 : matrix_fragment_9;

    matrix_fragment_60 = !condition_0 ? matrix_fragment_10 : matrix_fragment_59;

    matrix_fragment_61 = !condition_0 ? matrix_fragment_57 : matrix_fragment_58;

    matrix_fragment_10 = !condition_0 ? matrix_fragment_59 : matrix_fragment_10;

    matrix_fragment_57 = !condition_0 ? matrix_fragment_58 : matrix_fragment_57;

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_60 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_60),
                                        select_half_operand<1, false, false>(matrix_fragment_60));

    matrix_fragment_60 = permute_signed(matrix_fragment_60, matrix_fragment_60, 0x5410u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x4100410u);

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_60 & 65535u) * 65537u);

    matrix_fragment_60 = half_to_float_bits((matrix_fragment_60 >> 16) * 65537u);

    matrix_fragment_8 = reciprocal_bits(matrix_fragment_8);

    matrix_fragment_9 = reciprocal_bits(matrix_fragment_60);

    matrix_accumulator_120 =
        uint32_t(combine_words(matrix_accumulator_120, 0u) << 0x5u >> 32) + matrix_fragment_52;

    matrix_fragment_11 = matrix_accumulator_120 & 0xffffffe0u;

    matrix_fragment_32 = uint32_t(int64_t(int32_t(index_word_164)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_57 = uint32_t(int32_t(matrix_fragment_32) >> 0x1fu);

    matrix_fragment_10 = pack_half(matrix_fragment_9, matrix_fragment_8);

    matrix_fragment_9 = matrix_fragment_52 + (0u - matrix_fragment_11);

    matrix_fragment_11 = uint32_t(int64_t(int32_t(index_word_164)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_57 =
        uint32_t(combine_words(matrix_fragment_57, 0u) << 0x5u >> 32) + matrix_fragment_32;

    matrix_fragment_8 = uint32_t(int32_t(matrix_fragment_11) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_fragment_59 = matrix_fragment_57 & 0xffffffe0u;

    matrix_fragment_8 =
        uint32_t(combine_words(matrix_fragment_8, 0u) << 0x5u >> 32) + matrix_fragment_11;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_53), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    matrix_fragment_8 = matrix_fragment_8 & 0xffffffe0u;

    matrix_fragment_59 = matrix_fragment_32 + (0u - matrix_fragment_59);

    matrix_fragment_57 = matrix_fragment_11 + (0u - matrix_fragment_8);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_59), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_59 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_57), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    matrix_fragment_58 = permute_signed(matrix_fragment_9, matrix_fragment_9, 0x5410u);

    matrix_fragment_32 = permute_signed(matrix_fragment_53, matrix_fragment_53, 0x5410u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_51),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_49),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_49 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_28),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_59 = permute_signed(matrix_fragment_59, matrix_fragment_59, 0x5410u);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_accumulator_49));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_12 = permute_signed(matrix_fragment_56, matrix_fragment_56, 0x5410u);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_29),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_28));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_31),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_48),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_29 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_29),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_50),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_31),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_30),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_48 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_4 = parameters[parameter_trained_weights_low_4];
    matrix_fragment_5 = parameters[parameter_trained_weights_high_5];

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_20, matrix_accumulator_21, 0u, 0u);

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                      select_half_operand<2, false, false>(matrix_accumulator_29));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_48));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_94, matrix_fragment_95, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_30),
                                      select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_94, matrix_fragment_95, matrix_fragment_46,
                             matrix_fragment_47);

    packed_half_0 = index_word_164 & 0x7ffffffcu;

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_44,
                             matrix_fragment_45);

    packed_half_0 = 0u - packed_half_0 + address_word_159;

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_92, matrix_fragment_93, matrix_accumulator_22,
                             matrix_accumulator_23);

    index_word_164 = index_word_164 & 0xfffffffcu;

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_20,
                             matrix_accumulator_21);

    matrix_fragment_35 = packed_half_0 * 0x2u + matrix_fragment_35;

    matrix_accumulator_29 = matrix_fragment_7 + address_word_159 + (0u - index_word_164);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_31 = matrix_fragment_33 + address_word_159 + (0u - index_word_164);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_35) >> 0x1u);

    matrix_fragment_11 = matrix_fragment_34 + address_word_159 + (0u - index_word_164);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    split_address_words(matrix_accumulator_28, matrix_accumulator_29,
                        uint64_t(uint32_t(matrix_accumulator_29)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                  select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_6 + 0x800u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_30, matrix_accumulator_31,
                        uint64_t(uint32_t(matrix_accumulator_31)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_6, matrix_fragment_52,
                                         matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(packed_half_3));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_37 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_28, matrix_accumulator_29) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    matrix_fragment_36 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0xf0b0ull, memory_ranges,
        error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_6 + 0x400u, matrix_fragment_40, matrix_fragment_41,
        matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_38 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_30, matrix_accumulator_31) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0xf0b0ull, memory_ranges,
        error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_6 + 0xc00u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe0b0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe2b0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xe8b0ull, matrix_accumulator_16,
        matrix_accumulator_17, matrix_accumulator_18, matrix_accumulator_19, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_82, matrix_fragment_83) + 0xeab0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    uniform_index_word_5 = 0u;

    uniform_index_word_4 = 0x400u;

    uniform_index_word_4 = (uniform_index_word_5 << 0x18u) + uniform_index_word_4;

    matrix_accumulator_50 = (address_word_159 << 0x4u) + uniform_index_word_4;

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_50, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_0 = unpack_fp8_bits(matrix_fragment_32);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_0 = unpack_fp8_bits(matrix_fragment_34);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_35);

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    packed_half_0 = unpack_fp8_bits(matrix_accumulator_20);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    packed_half_3 = unpack_fp8_bits(matrix_accumulator_21);

    matrix_accumulator_20 = unpack_fp8_bits(matrix_accumulator_20 >> 16);

    matrix_accumulator_21 = unpack_fp8_bits(matrix_accumulator_21 >> 16);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_50 + 0x400u, matrix_fragment_4, matrix_fragment_5,
        matrix_fragment_6, matrix_fragment_7, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_34 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_28,
                             matrix_accumulator_29);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_0 = unpack_fp8_bits(matrix_accumulator_22);

    matrix_accumulator_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_3 = unpack_fp8_bits(matrix_accumulator_23);

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_22 >> 16);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_23 >> 16);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_20,
                             matrix_accumulator_21);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_34,
                             matrix_fragment_35);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_50 + 0x600u, matrix_fragment_32, matrix_fragment_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    packed_half_3 = threadIdx.z;

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_28,
                             matrix_accumulator_29);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_50 + 0x200u, matrix_accumulator_28, matrix_accumulator_29,
        matrix_accumulator_30, matrix_accumulator_31, shared_bytes, memory_ranges, error_counts);

    packed_half_0 = index_word_151 + 0x1u;

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_20,
                             matrix_accumulator_21);

    condition_0 = int32_t(packed_half_0) >= int32_t(index_word_157);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_22,
                             matrix_accumulator_23);

    matrix_fragment_6 = threadIdx.x;

    matrix_fragment_4 = parameters[parameter_output_low_2];
    matrix_fragment_5 = parameters[parameter_output_high_3];

    condition_0 = int32_t(matrix_fragment_74) > int32_t(0u - 0x8u) && !condition_0;

    index_word_151 = index_word_151 * index_word_155 + index_word_155;

    condition_5 = condition_0 & condition_5;

    index_word_151 = index_word_156 + index_word_151;

    condition_0 = condition_0 & condition_4;

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_44,
                             matrix_fragment_45);

    condition_1 = (matrix_fragment_6 | packed_half_3 | thread_row_165) != 0;

    packed_half_3 = index_word_151 * 0x100u + packed_half_2;

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_24,
                             matrix_fragment_25);

    split_address_words(packed_half_2, packed_half_3,
                        int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_4, matrix_fragment_5));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_accumulator_48, matrix_accumulator_49, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_48,
                             matrix_accumulator_49);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_accumulator_28,
                             matrix_accumulator_29, matrix_accumulator_30, matrix_accumulator_31,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_40,
                             matrix_fragment_41);
    if (condition_5) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                          select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_49),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                          select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_48),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(0u));
    if (condition_0) {
        matrix_accumulator_21 = permute_signed(matrix_accumulator_21, matrix_fragment_4, 0x5410u);

        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(address_word_159)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        packed_half_0 = permute_signed(matrix_fragment_25, packed_half_0, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_accumulator_20, matrix_fragment_7, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_24, matrix_fragment_5, 0x5410u);

        matrix_fragment_5 = packed_half_0;

        matrix_fragment_7 = matrix_accumulator_21;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, packed_half_3) + 0x400ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_1) {
        return;
    }

    uniform_index_word_4 = blockIdx.x;

    packed_half_2 = parameters[parameter_publication_optional_low_16];
    packed_half_3 = parameters[parameter_publication_optional_high_17];

    matrix_fragment_5 = reused_word_162 + uniform_index_word_4;

    split_address_words(packed_half_2, packed_half_3,
                        int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_2, packed_half_3));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(packed_half_2, packed_half_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__
__maxnreg__(128) void phoenix_window_attention_channels_64_attention(KernelArguments parameters) {
    run_window_attention_channels_64_attention<false>(parameters.words, nullptr, nullptr);
}
