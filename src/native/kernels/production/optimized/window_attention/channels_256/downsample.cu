// DLSS5 Phoenix optimized kernel: window_attention/channels_256/downsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#include "../../../common/boundary_clear.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 17408
#include "../../../common/device_memory_vector_fp8.cuh"
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
    parameter_output_downsample_optional_low_18 = 18,
    parameter_output_downsample_optional_high_19 = 19,
    parameter_tensor_height_20 = 20,
    parameter_tensor_width_21 = 21,
};

struct alignas(8) KernelArguments {
    uint32_t words[22];
};
static_assert(sizeof(KernelArguments) == 88, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_256_downsample(
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
    uint32_t packed_half_10 = 0;
    uint32_t matrix_fragment_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t matrix_accumulator_16 = 0;
    uint32_t matrix_accumulator_17 = 0;
    uint32_t matrix_fragment_18 = 0;
    uint32_t matrix_fragment_19 = 0;
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_accumulator_22 = 0;
    uint32_t matrix_accumulator_23 = 0;
    uint32_t matrix_fragment_24 = 0;
    uint32_t matrix_fragment_25 = 0;
    uint32_t matrix_fragment_26 = 0;
    uint32_t matrix_fragment_27 = 0;
    uint32_t matrix_fragment_28 = 0;
    uint32_t matrix_fragment_29 = 0;
    uint32_t matrix_fragment_30 = 0;
    uint32_t matrix_fragment_31 = 0;
    uint32_t matrix_accumulator_32 = 0;
    uint32_t matrix_accumulator_33 = 0;
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
    uint32_t matrix_fragment_104 = 0;
    uint32_t matrix_fragment_105 = 0;
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
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t packed_half_144 = 0;
    uint32_t address_word_145 = 0;
    uint32_t address_word_146 = 0;
    uint32_t address_word_147 = 0;
    uint32_t index_word_148 = 0;
    uint32_t tile_word_149 = 0;
    uint32_t index_word_150 = 0;
    uint32_t index_word_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t index_word_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t index_word_155 = 0;
    uint32_t thread_row_156 = 0;
    uint32_t address_word_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t address_word_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_parameter_word_7 = 0;
    uint32_t uniform_parameter_word_8 = 0;
    uint32_t uniform_parameter_word_9 = 0;
    uint32_t uniform_reused_word_10 = 0;
    uint32_t uniform_reused_word_11 = 0;
    uint32_t uniform_parameter_word_12 = 0;
    uint32_t uniform_reused_word_13 = 0;
    uint32_t uniform_block_index_14 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_address_word_17 = 0;
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

    uniform_parameter_word_9 = parameters[parameter_optional_control_15];

    matrix_fragment_8 = parameters[parameter_optional_control_10];
    matrix_fragment_9 = parameters[parameter_scalar_control_11];

    matrix_accumulator_3 = blockIdx.x;

    uniform_parameter_word_7 = parameters[parameter_tensor_height_8];

    matrix_fragment_5 = threadIdx.x;

    uniform_parameter_word_8 = parameters[parameter_scalar_control_14];

    matrix_accumulator_138 = threadIdx.y;

    uniform_index_word_6 = parameters[parameter_tensor_width_9];

    matrix_fragment_6 = threadIdx.z;

    uniform_reused_word_10 = 0u;
    uniform_reused_word_11 = 0u;

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_parameter_word_9));

    uniform_parameter_word_7 = 0u - uniform_index_word_4 + uniform_parameter_word_7;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_parameter_word_8));

    uniform_index_word_4 = 0u;

    uniform_condition_1 = uint32_t(uniform_parameter_word_7) != 0u;

    uniform_index_word_6 = 0u - uniform_index_word_5 + uniform_index_word_6;

    packed_half_0 = packed_half_0 * 0x8u + matrix_fragment_9;

    uniform_index_word_5 = 0u;

    uniform_condition_0 = uint32_t(uniform_index_word_6) != 0u;

    matrix_fragment_4 = matrix_accumulator_3 * 0x8u + matrix_fragment_8;

    matrix_accumulator_2 = packed_half_0 + (0u - uniform_parameter_word_9);

    if (uniform_condition_1) {
        uniform_parameter_word_7 = uniform_parameter_word_7 + 0xffffffffu;
    }

    matrix_fragment_7 = matrix_fragment_4 + (0u - uniform_parameter_word_8);

    matrix_fragment_8 = matrix_accumulator_2 + 0x8u;

    matrix_accumulator_3 = matrix_accumulator_2 + 0x7u;

    if (uniform_condition_1) {
        uniform_parameter_word_8 = uint32_t(int32_t(uniform_parameter_word_7) >> 0x1fu);
    }

    matrix_accumulator_2 = max(int32_t(0u), int32_t(matrix_accumulator_2));

    if (uniform_condition_0) {
        uniform_index_word_6 = uniform_index_word_6 + 0xffffffffu;
    }

    condition_1 = int32_t(matrix_fragment_8) > int32_t(0u);

    condition_0 = (matrix_accumulator_138 | matrix_fragment_5 | matrix_fragment_6) != 0;

    if (uniform_condition_1) {
        uniform_parameter_word_8 =
            uint32_t(combine_words(uniform_parameter_word_8, 0u) << 0x3u >> 32) +
            uniform_parameter_word_7;
    }

    matrix_fragment_5 = uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_parameter_word_7 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);
    }

    matrix_accumulator_2 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

    if (uniform_condition_1) {
        uniform_index_word_5 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_8)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_fragment_9 = matrix_fragment_7 + 0x8u;

    matrix_fragment_6 = condition_1 ? matrix_fragment_5 : 0u;

    matrix_fragment_8 = matrix_fragment_7 + 0x7u;

    matrix_accumulator_3 = uint32_t(int32_t(packed_half_0) >> 0x1fu);

    if (uniform_condition_0) {
        uniform_parameter_word_7 =
            uint32_t(combine_words(uniform_parameter_word_7, 0u) << 0x3u >> 32) +
            uniform_index_word_6;
    }

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    packed_half_10 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_5));

    condition_2 = int32_t(matrix_fragment_9) > int32_t(0u);

    matrix_fragment_8 = uint32_t(combine_words(matrix_fragment_8, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_0) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_parameter_word_7)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + packed_half_0;

    condition_0 = int32_t(matrix_accumulator_2) >= int32_t(packed_half_10) || condition_0;

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_4;

    matrix_fragment_7 = max(int32_t(0u), int32_t(matrix_fragment_7));

    index_word_155 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    index_word_153 = uint32_t(int32_t(matrix_fragment_5) >> 0x2u);

    matrix_fragment_8 = condition_2 ? matrix_fragment_8 : 0u;
    if (!condition_0) {
        matrix_fragment_7 = uint32_t(uint32_t(matrix_fragment_7) >> 0x3u);

        matrix_fragment_8 = min(int32_t(matrix_fragment_8), int32_t(uniform_index_word_4));

        matrix_fragment_5 = matrix_accumulator_2;
        do {
            condition_0 = int32_t(matrix_fragment_7) >= int32_t(matrix_fragment_8);
            if (!condition_0) {
                matrix_fragment_11 = matrix_fragment_5 * uniform_index_word_4;

                matrix_fragment_6 = matrix_fragment_7;
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_12];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_9 = matrix_fragment_11 + matrix_fragment_6;

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

            condition_0 = uint32_t(matrix_fragment_5) != uint32_t(packed_half_10);
        } while (condition_0);
    }
    matrix_fragment_34 = parameters[parameter_tensor_height_8];
    matrix_fragment_35 = parameters[parameter_tensor_width_9];

    condition_3 = false;

    address_word_145 = threadIdx.x & 31u;

    condition_5 = false;

    __syncwarp(__activemask());

    matrix_fragment_13 = index_word_153 + 0x1u;

    matrix_fragment_12 = 0u;

    matrix_fragment_14 = 0u;

    matrix_accumulator_16 = 0u;

    packed_half_10 = 0u;

    matrix_accumulator_2 = matrix_fragment_34 & 0xfffffffcu;

    matrix_accumulator_3 = uint32_t(int32_t(matrix_fragment_34) >> 0x1fu);

    matrix_fragment_5 = matrix_fragment_35 & 0xfffffffcu;

    condition_0 = uint32_t(matrix_accumulator_2) != uint32_t(0x4u);

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + matrix_fragment_34;

    matrix_accumulator_2 = uint32_t(int32_t(matrix_fragment_35) >> 0x1fu);

    index_word_151 = uint32_t(int32_t(matrix_accumulator_3) >> 0x2u);

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x2u >> 32) + matrix_fragment_35;

    index_word_150 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    if (condition_0) {
        matrix_fragment_6 = index_word_155 + 0x1u;

        matrix_fragment_8 = index_word_155 + 0x1u;

        condition_4 = int32_t(index_word_155) >= int32_t(index_word_151);

        condition_6 = int32_t(index_word_155) >= int32_t(index_word_151);

        matrix_fragment_12 = index_word_155 * index_word_150;

        condition_1 = int32_t(matrix_fragment_6) >= int32_t(index_word_151);

        condition_2 = int32_t(matrix_fragment_8) >= int32_t(index_word_151);

        condition_3 = int32_t(packed_half_0) < int32_t(0u - 0x3u) || condition_4;
    }

    condition_4 = false;

    if (condition_0) {
        matrix_fragment_14 = index_word_155 * index_word_150 + index_word_150;

        condition_5 = int32_t(packed_half_0) < int32_t(0u - 0x3u) || condition_6;
    }

    condition_6 = int32_t(index_word_153) >= int32_t(index_word_150);

    if (condition_0) {
        condition_4 = int32_t(packed_half_0) < int32_t(0u - 0x7u) || condition_1;
    }

    condition_1 = false;

    if (condition_0) {
        matrix_accumulator_16 = index_word_155 * index_word_150 + index_word_150;

        condition_1 = int32_t(packed_half_0) < int32_t(0u - 0x7u) || condition_2;
    }

    condition_2 = true;

    if (condition_0) {
        packed_half_10 = index_word_155 * index_word_150;

        condition_2 = !condition_3;
    }

    matrix_accumulator_2 = condition_3 ? index_word_153 : 0u;

    condition_6 = int32_t(matrix_fragment_4) > int32_t(0u - 0x4u) && !condition_6;

    condition_3 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_3;

    matrix_fragment_6 = condition_4 ? index_word_153 : 0u;

    packed_half_144 = condition_6 ? 0x40u : 0u;

    condition_2 = condition_2 & (condition_3 | condition_6);

    condition_3 = int32_t(matrix_fragment_13) >= int32_t(index_word_150);

    matrix_accumulator_3 = uint32_t(condition_2) << 2;

    condition_2 = int32_t(matrix_fragment_4) > int32_t(0u - 0x8u) && !condition_3;

    condition_3 = true;

    if (condition_0) {
        condition_3 = !condition_5;
    }

    matrix_fragment_4 = condition_5 ? matrix_fragment_13 : 0u;

    condition_5 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_5;

    packed_half_0 = uint32_t(condition_2) << 2;

    condition_3 = condition_3 & (condition_5 | condition_2);

    condition_5 = true;

    if (condition_0) {
        condition_5 = !condition_4;
    }

    condition_4 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_4;

    condition_4 = condition_5 & (condition_4 | condition_6);

    condition_5 = true;

    condition_6 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_1;

    if (condition_0) {
        condition_5 = !condition_1;
    }

    if (!condition_3) {
        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    condition_5 = condition_5 & (condition_6 | condition_2);

    condition_2 = int32_t(matrix_accumulator_3) != int32_t(0u);

    condition_6 = uint32_t(matrix_fragment_5) != uint32_t(0x4u);

    if (!condition_4) {
        matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (!condition_3) {
        matrix_fragment_19 = permute_signed(matrix_fragment_19, matrix_fragment_19, 0x5410u);
    }

    matrix_accumulator_17 = !condition_6 ? matrix_fragment_6 : index_word_153;

    matrix_fragment_15 = !condition_6 ? matrix_fragment_4 : matrix_fragment_13;

    if (condition_4) {
        matrix_fragment_6 = parameters[parameter_input_low_0];
        matrix_fragment_7 = parameters[parameter_input_high_1];
    }

    matrix_fragment_11 = !condition_6 ? matrix_accumulator_2 : index_word_153;

    if (!condition_4) {
        matrix_accumulator_23 =
            permute_signed(matrix_accumulator_23, matrix_accumulator_23, 0x5410u);
    }

    if (condition_3) {
        matrix_fragment_15 = matrix_fragment_15 + matrix_fragment_12;
    }

    if (condition_4) {
        matrix_accumulator_17 = matrix_accumulator_17 + matrix_fragment_14;
    }

    if (!condition_6) {
        matrix_fragment_13 = condition_1 ? matrix_fragment_13 : 0u;
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 + packed_half_10;

        matrix_accumulator_2 = parameters[parameter_input_low_0];
        matrix_accumulator_3 = parameters[parameter_input_high_1];
    }

    if (condition_3) {
        matrix_fragment_15 = matrix_fragment_15 * 0x8u + matrix_accumulator_138;
    }

    if (!condition_5) {
        matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    if (condition_4) {
        matrix_accumulator_17 = matrix_accumulator_17 * 0x8u + matrix_accumulator_138;
    }

    if (condition_5) {
        matrix_fragment_13 = matrix_fragment_13 + matrix_accumulator_16;
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 * 0x8u + matrix_accumulator_138;
    }

    if (!condition_5) {
        matrix_fragment_39 = permute_signed(matrix_fragment_39, matrix_fragment_39, 0x5410u);
    }

    if (condition_3) {
        matrix_fragment_4 = parameters[parameter_input_low_0];
        matrix_fragment_5 = parameters[parameter_input_high_1];
    }

    if (condition_5) {
        packed_half_10 = matrix_fragment_13 * 0x8u + matrix_accumulator_138;
    }

    if (!condition_3) {
        matrix_accumulator_16 = matrix_fragment_19;
    }

    if (condition_3) {
        matrix_fragment_13 = matrix_fragment_15 * 0x80u;
    }

    if (!condition_3) {
        matrix_fragment_18 = matrix_fragment_19;
    }

    if (condition_4) {
        matrix_fragment_15 = matrix_accumulator_17 * 0x80u;
    }

    if (!condition_4) {
        matrix_accumulator_20 = matrix_accumulator_23;
    }

    if (condition_2) {
        matrix_fragment_11 = matrix_fragment_11 * 0x80u;
    }

    if (condition_5) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];
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
        matrix_accumulator_21 = matrix_accumulator_23;
    }

    if (condition_5) {
        matrix_accumulator_17 = packed_half_10 * 0x80u;
    }

    if (!condition_2) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_4) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_6, matrix_fragment_7));
    }

    if (!condition_4) {
        matrix_accumulator_22 = matrix_accumulator_23;
    }

    __syncthreads();

    if (!condition_5) {
        matrix_fragment_36 = matrix_fragment_39;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (!condition_2) {
        matrix_fragment_12 = matrix_fragment_15;

        matrix_fragment_14 = matrix_fragment_15;
    }

    if (!condition_5) {
        matrix_fragment_37 = matrix_fragment_39;

        matrix_fragment_38 = matrix_fragment_39;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (!condition_2) {
        matrix_fragment_13 = matrix_fragment_15;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (condition_5) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_3) {
        matrix_accumulator_17 = matrix_fragment_19;
    }

    if (condition_3) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));
    }

    if (condition_5) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
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
            matrix_accumulator_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
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
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);
    }

    uniform_index_word_6 = 0u;

    uniform_index_word_5 = 0x400u;

    uniform_parameter_word_8 = parameters[parameter_trained_weights_low_4];
    uniform_parameter_word_9 = parameters[parameter_trained_weights_high_5];

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                        combine_words(uniform_parameter_word_8, uniform_parameter_word_9));

    uniform_index_word_6 = (uniform_index_word_6 << 0x18u) + uniform_index_word_5;

    uniform_index_word_5 = 0u;

    packed_half_10 = (matrix_accumulator_138 << 0x9u) + uniform_index_word_6;

    tile_word_149 = address_word_145 * 0x10u + packed_half_10;

    packed_half_10 = 0;
    matrix_fragment_11 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149 + 0x1000u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149 + 0x2000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x3000u, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_accumulator_16 = 0;
    matrix_accumulator_17 = 0;

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    __syncthreads();
    do {
        split_address_words(address_word_146, address_word_147,
                            uint64_t(uint32_t(matrix_accumulator_138)) *
                                    uint64_t(uint32_t(0x8000u)) +
                                combine_words(matrix_accumulator_142, matrix_accumulator_143));

        split_address_words(address_word_146, address_word_147,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_146, address_word_147));

        load_four_words_cached<CheckBounds>(combine_words(address_word_146, address_word_147),
                                            matrix_fragment_92, matrix_fragment_93,
                                            matrix_fragment_94, matrix_fragment_95, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x200ull, matrix_fragment_96,
            matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x1000ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x1200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x2000ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x2200ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x3000ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x3200ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        matrix_accumulator_139 = (address_word_145 << 0x4u) + uniform_index_word_6;

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x4000ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_139,
                                            matrix_accumulator_112, matrix_accumulator_113,
                                            matrix_accumulator_114, matrix_accumulator_115,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x4200ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1000u, matrix_fragment_72, matrix_fragment_73,
            matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x5000ull, matrix_fragment_68,
            matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2000u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_139 + 0x3000u,
                                            matrix_accumulator_100, matrix_accumulator_101,
                                            matrix_accumulator_102, matrix_accumulator_103,
                                            shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x200u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1200u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3200u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_accumulator_112,
                                 matrix_accumulator_113, matrix_accumulator_114,
                                 matrix_accumulator_115, matrix_fragment_94, matrix_fragment_95, 0u,
                                 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_96,
                                 matrix_fragment_97, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_98,
                                 matrix_fragment_99, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_96, matrix_fragment_97, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_98, matrix_fragment_99, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x5200ull, matrix_fragment_72,
            matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_92, matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_94, matrix_fragment_95, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_96, matrix_fragment_97, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_98, matrix_fragment_99, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2200u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_92,
                                 matrix_fragment_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_94,
                                 matrix_fragment_95, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x6000ull, matrix_fragment_92,
            matrix_fragment_93, matrix_fragment_94, matrix_fragment_95, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_96,
                                 matrix_fragment_97, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_98,
                                 matrix_fragment_99, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x6200ull, matrix_fragment_96,
            matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges,
            error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_139 + 0x400u,
                                            matrix_accumulator_100, matrix_accumulator_101,
                                            matrix_accumulator_102, matrix_accumulator_103,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_104,
                                 matrix_fragment_105);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_139 + 0x1400u,
                                            matrix_fragment_104, matrix_fragment_105,
                                            matrix_accumulator_106, matrix_accumulator_107,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x7000ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x600u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_122,
                                 matrix_accumulator_123);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3400u, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2400u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_46,
                                 matrix_fragment_47, matrix_fragment_36, matrix_fragment_37);

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_146, address_word_147) + 0x7200ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        multiply_accumulate_half(
            matrix_accumulator_134, matrix_accumulator_135, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_44, matrix_fragment_45, matrix_accumulator_134, matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_44, matrix_fragment_45,
                                 matrix_accumulator_132, matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_76, matrix_fragment_77);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_116,
                                 matrix_accumulator_117);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3600u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(
            matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_48, matrix_fragment_49, matrix_accumulator_136, matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129,
                                 matrix_accumulator_100, matrix_accumulator_101,
                                 matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_50,
                                 matrix_fragment_51, matrix_fragment_78, matrix_fragment_79);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1600u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_139 + 0x2600u,
                                            matrix_accumulator_100, matrix_accumulator_101,
                                            matrix_accumulator_102, matrix_accumulator_103,
                                            shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_48, matrix_fragment_49,
                                 matrix_accumulator_112, matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_118, matrix_accumulator_119);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x800u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3800u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2800u, matrix_fragment_80, matrix_fragment_81,
            matrix_fragment_82, matrix_fragment_83, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0xa00u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1800u, matrix_fragment_76, matrix_fragment_77,
            matrix_fragment_78, matrix_fragment_79, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3a00u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_108, matrix_accumulator_109, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_52, matrix_fragment_53, matrix_accumulator_108, matrix_accumulator_109);

        multiply_accumulate_half(
            matrix_accumulator_114, matrix_accumulator_115, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_54, matrix_fragment_55, matrix_accumulator_114, matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1a00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(
            matrix_accumulator_120, matrix_accumulator_121, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_56, matrix_fragment_57, matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(
            matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_100,
            matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103,
            matrix_fragment_58, matrix_fragment_59, matrix_accumulator_124, matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2a00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_128,
                                 matrix_accumulator_129);

        split_address_words(matrix_fragment_48, matrix_fragment_49,
                            uint64_t(uint32_t(matrix_accumulator_138)) *
                                    uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_accumulator_142, matrix_accumulator_143));

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_76,
                                 matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_118,
                                 matrix_accumulator_119);

        split_address_words(matrix_fragment_76, matrix_fragment_77,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_48, matrix_fragment_49));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_76, matrix_fragment_77) + 0x40000ull, matrix_fragment_48,
            matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_130,
                                 matrix_accumulator_131);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_76, matrix_fragment_77) + 0x40200ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_114,
                                 matrix_accumulator_115);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2c00u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_80,
                                 matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1e00u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x1c00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0xc00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_96, matrix_fragment_97, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_98, matrix_fragment_99, matrix_accumulator_118,
                                 matrix_accumulator_119);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0xe00u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_96, matrix_fragment_97, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_98, matrix_fragment_99, matrix_accumulator_128,
                                 matrix_accumulator_129);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3c00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_96, matrix_fragment_97, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_98, matrix_fragment_99, matrix_accumulator_124,
                                 matrix_accumulator_125);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x2e00u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_92, matrix_fragment_93, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_94, matrix_fragment_95, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_96, matrix_fragment_97, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_98, matrix_fragment_99, matrix_accumulator_130,
                                 matrix_accumulator_131);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_122), 0x44004400u);

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        matrix_fragment_53 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_132,
                                 matrix_accumulator_133);

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_123), 0x44004400u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_118,
                                 matrix_accumulator_119);

        matrix_fragment_66 = 0x2b28u;

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_128,
                                 matrix_accumulator_129);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_139 + 0x3e00u, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_67 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_136), 0x44004400u);

        matrix_accumulator_122 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_137), 0x44004400u);

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_fragment_69 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_128), 0x44004400u);

        matrix_fragment_72 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

        matrix_fragment_69 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_114,
                                 matrix_accumulator_115);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_120,
                                 matrix_accumulator_121);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_70 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_129), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_124,
                                 matrix_accumulator_125);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_132), 0x44004400u);

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_140), 0x44004400u);

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_133), 0x44004400u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_73 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_141), 0x44004400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_69),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_40, matrix_fragment_41, matrix_accumulator_110,
                                 matrix_accumulator_111);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

        matrix_accumulator_136 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_112), 0x44004400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_41 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_accumulator_137 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_70 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_67 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_113), 0x44004400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_42, matrix_fragment_43, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_41),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_69 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

        matrix_accumulator_141 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_accumulator_123 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_72),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x44004400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_37 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_72 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        matrix_accumulator_128 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_98,
                                 matrix_fragment_99);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_63 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_69),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_37),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_59 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                select_half_operand<2, false, false>(matrix_fragment_36));

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_110), 0x44004400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0xc400c400u);

        matrix_fragment_71 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69),
            select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

        matrix_fragment_38 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37),
            select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_111), 0x44004400u);

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_71 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_73),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73),
            select_half_operand<2, false, false>(matrix_fragment_74), 0x3b283b28u);

        matrix_fragment_64 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_64), 0x3b283b28u);

        matrix_fragment_67 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_124), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0x44004400u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_69 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_125), 0x44004400u);

        matrix_accumulator_129 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                select_half_operand<2, false, false>(matrix_fragment_74));

        matrix_accumulator_140 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_68 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_64 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                select_half_operand<2, false, false>(matrix_fragment_64));

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_69), 0xc400c400u);

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_72),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_68),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_36 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_37 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_73 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72),
            select_half_operand<2, false, false>(matrix_fragment_73), 0x3b283b28u);

        matrix_fragment_38 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                          select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_39 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                          select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_69 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68),
            select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0x44004400u);

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

        matrix_fragment_53 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_fragment_68 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99), 0x44004400u);

        matrix_fragment_65 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

        matrix_fragment_70 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_68), 0xc400c400u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_70),
            select_half_operand<0, false, true>(matrix_fragment_66), 0x37283728u);

        matrix_fragment_68 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_68), 0x3b283b28u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_70),
            select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

        matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_69 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_68 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                select_half_operand<2, false, false>(matrix_fragment_68));

        matrix_fragment_72 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                select_half_operand<2, false, false>(matrix_fragment_72));

        matrix_fragment_58 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_57 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_110 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                select_half_operand<2, false, false>(matrix_fragment_55));

        matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_65 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                                select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                          select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_62 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                          select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                          select_half_operand<2, false, false>(matrix_fragment_74));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                          select_half_operand<2, false, false>(matrix_fragment_72));

        matrix_fragment_60 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                          select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_61 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_63 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                          select_half_operand<2, false, false>(matrix_fragment_66));

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_32,
                                 matrix_accumulator_33);

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_4 = uniform_index_word_5 + 0x20u;

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_48, matrix_fragment_49, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(packed_half_10, matrix_fragment_11, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_46, matrix_fragment_47, packed_half_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_2,
                                 matrix_accumulator_3);

        split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                            combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x400u);

        uniform_index_word_5 = uniform_index_word_4;
    } while (!uniform_condition_0);
    address_word_158 = parameters[parameter_trained_weights_low_4];
    address_word_159 = parameters[parameter_trained_weights_high_5];

    index_word_148 = matrix_accumulator_138 * 0x20u;

    address_word_147 = address_word_145;

    matrix_fragment_40 = uint32_t(uint32_t(index_word_148) >> 0x1u);

    address_word_146 = address_word_147 & 0xfffffffcu;

    matrix_fragment_36 = matrix_fragment_40 + 0x4u;

    matrix_fragment_38 = matrix_fragment_40 + 0x8u;

    matrix_fragment_41 = matrix_fragment_40 + 0xcu;

    matrix_fragment_40 = address_word_147 & 0x7ffffffcu;

    matrix_fragment_37 = matrix_fragment_36 + address_word_145 + (0u - address_word_146);

    matrix_fragment_45 = matrix_fragment_41 + address_word_145 + (0u - address_word_146);

    matrix_fragment_41 = 0u - matrix_fragment_40 + address_word_145;

    matrix_fragment_39 = matrix_fragment_38 + address_word_145 + (0u - address_word_146);

    split_address_words(matrix_fragment_36, matrix_fragment_37,
                        uint64_t(uint32_t(matrix_fragment_37)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_41 = matrix_fragment_41 * 0x2u + index_word_148;

    matrix_fragment_53 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_36, matrix_fragment_37) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_38, matrix_fragment_39,
                        uint64_t(uint32_t(matrix_fragment_39)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_41 = uint32_t(int32_t(matrix_fragment_41) >> 0x1u);

    matrix_fragment_54 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_38, matrix_fragment_39) + 0x58010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_44, matrix_fragment_45,
                        uint64_t(uint32_t(matrix_fragment_45)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(packed_half_160, address_word_161,
                        int64_t(int32_t(matrix_fragment_41)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_56 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_44, matrix_fragment_45) + 0x58010ull, memory_ranges,
        error_counts);

    matrix_fragment_52 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, address_word_161) + 0x58010ull, memory_ranges, error_counts);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_4 = 0u;

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_149, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(0u));

    uniform_index_word_5 = uniform_index_word_6;

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x1000u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                   select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149 + 0x2000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(packed_half_10));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x3000u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x1000u, matrix_fragment_48,
                                         matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(packed_half_10));

    packed_half_10 = unpack_fp8_bits(matrix_fragment_42);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149 + 0x2000u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_43);

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_40);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x3000u, matrix_fragment_60,
                                         matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_40 >> 16);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_41);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149, matrix_fragment_36,
                                         matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_41 >> 16);

    matrix_fragment_6 = unpack_fp8_bits(matrix_accumulator_20 >> 16);

    matrix_fragment_7 = unpack_fp8_bits(matrix_accumulator_21 >> 16);

    matrix_fragment_27 = unpack_fp8_bits(matrix_accumulator_23);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_fragment_46);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_fragment_47);

    matrix_fragment_19 = unpack_fp8_bits(matrix_accumulator_20);

    matrix_accumulator_20 = unpack_fp8_bits(matrix_accumulator_21);

    matrix_accumulator_21 = unpack_fp8_bits(matrix_accumulator_22);

    matrix_accumulator_22 = unpack_fp8_bits(matrix_accumulator_22 >> 16);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_44);

    matrix_accumulator_23 = unpack_fp8_bits(matrix_accumulator_23 >> 16);

    matrix_accumulator_32 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_42 = unpack_fp8_bits(matrix_fragment_42 >> 16);

    matrix_fragment_43 = unpack_fp8_bits(matrix_fragment_43 >> 16);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_44 >> 16);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_45);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_45 >> 16);

    matrix_fragment_46 = unpack_fp8_bits(matrix_fragment_46 >> 16);

    matrix_fragment_47 = unpack_fp8_bits(matrix_fragment_47 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_accumulator_33 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                        uint64_t(uint32_t(matrix_accumulator_138)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x48200u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_32),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    __syncthreads();
    do {
        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_16, matrix_accumulator_17));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19) - 0x200ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_18, matrix_fragment_19),
                                            matrix_fragment_48, matrix_fragment_49,
                                            matrix_fragment_50, matrix_fragment_51, memory_ranges,
                                            error_counts);

        matrix_accumulator_20 = (address_word_145 << 0x4u) + uniform_index_word_5;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                            combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x2000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_20, matrix_fragment_52, matrix_fragment_53,
            matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_5 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_20 + 0x1000u, matrix_fragment_56, matrix_fragment_57,
            matrix_fragment_58, matrix_fragment_59, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_20 + 0x2000u, matrix_fragment_60, matrix_fragment_61,
            matrix_fragment_62, matrix_fragment_63, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_accumulator_20 + 0x3000u, matrix_fragment_64, matrix_fragment_65,
            matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_52,
                                 matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_36, matrix_fragment_37, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_38, matrix_fragment_39, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(packed_half_10, matrix_fragment_11, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_48, matrix_fragment_49, packed_half_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_56,
                                 matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_60,
                                 matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                 matrix_fragment_50, matrix_fragment_51, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_48, matrix_fragment_49, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_50, matrix_fragment_51, matrix_accumulator_32,
                                 matrix_accumulator_33);
    } while (!uniform_condition_0);
    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                          select_half_operand<2, false, false>(0u));

    __syncthreads();

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                        uint64_t(uint32_t(matrix_accumulator_138)) * uint64_t(uint32_t(0xc00u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = 0;
    matrix_fragment_31 = 0;

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_118 = 0;
    matrix_accumulator_119 = 0;

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                        combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x58c20u);

    matrix_accumulator_116 = 0;
    matrix_accumulator_117 = 0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = 0;
    matrix_accumulator_115 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 = 0;
    matrix_accumulator_113 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = 0;
    matrix_fragment_37 = 0;

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                      select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_38 = 0;
    matrix_fragment_39 = 0;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_40 = 0;
    matrix_fragment_41 = 0;

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_26 = 0;
    matrix_fragment_27 = 0;

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_42 = 0;
    matrix_fragment_43 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149, matrix_accumulator_16, matrix_accumulator_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_46 = 0;
    matrix_fragment_47 = 0;

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_48 = 0;
    matrix_fragment_49 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = 0;
    matrix_fragment_51 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 = 0;
    matrix_accumulator_127 = 0;

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_124 = 0;
    matrix_accumulator_125 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_122 = 0;
    matrix_accumulator_123 = 0;

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149 + 0x1000u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_24 = 0;
    matrix_fragment_25 = 0;

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = 0;
    matrix_fragment_19 = 0;

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_16 = 0;
    matrix_accumulator_17 = 0;

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = 0;
    matrix_fragment_45 = 0;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_120 = 0;
    matrix_accumulator_121 = 0;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_32 = 0;
    matrix_accumulator_33 = 0;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_52 = 0;
    matrix_fragment_53 = 0;

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(packed_half_10));

    matrix_accumulator_22 = 0;
    matrix_accumulator_23 = 0;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_20 = 0;
    matrix_accumulator_21 = 0;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x2000u, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_28 = 0;
    matrix_fragment_29 = 0;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_10 = 0;
    matrix_fragment_11 = 0;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_accumulator_2 = 0;
    matrix_accumulator_3 = 0;

    matrix_fragment_54 = 0;
    matrix_fragment_55 = 0;

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x3000u, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_56 = 0;
    matrix_fragment_57 = 0;

    matrix_fragment_58 = 0;
    matrix_fragment_59 = 0;

    matrix_fragment_104 = 0;
    matrix_fragment_105 = 0;

    matrix_accumulator_106 = 0;
    matrix_accumulator_107 = 0;

    matrix_accumulator_108 = 0;
    matrix_accumulator_109 = 0;

    matrix_accumulator_110 = 0;
    matrix_accumulator_111 = 0;

    matrix_accumulator_142 = 0;
    matrix_accumulator_143 = 0;

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

    matrix_fragment_14 = 0;
    matrix_fragment_15 = 0;

    matrix_fragment_12 = 0;
    matrix_fragment_13 = 0;

    matrix_accumulator_140 = 0;
    matrix_accumulator_141 = 0;

    uniform_index_word_4 = 0u;

    uniform_index_word_5 = uniform_index_word_6;

    __syncthreads();
    do {
        split_address_words(matrix_accumulator_102, matrix_accumulator_103,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
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

        packed_half_152 = (address_word_145 << 0x4u) + uniform_index_word_5;

        uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xe0u);

        uniform_parameter_word_7 = uniform_index_word_4 + 0x20u;

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            combine_words(matrix_accumulator_100, matrix_accumulator_101) +
                                0x6000u);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152, matrix_fragment_84, matrix_fragment_85,
            matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_5 = uniform_index_word_5 + 0x200u;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152 + 0x1000u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        uniform_index_word_4 = uniform_parameter_word_7;

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152 + 0x2000u, matrix_fragment_92, matrix_fragment_93,
            matrix_fragment_94, matrix_fragment_95, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, packed_half_152 + 0x3000u, matrix_fragment_96, matrix_fragment_97,
            matrix_fragment_98, matrix_fragment_99, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_140,
                                 matrix_accumulator_141);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_58,
                                 matrix_fragment_59);

        multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_42,
                                 matrix_fragment_43);

        multiply_accumulate_half(packed_half_10, matrix_fragment_11, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_60, matrix_fragment_61, packed_half_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_62, matrix_fragment_63, matrix_accumulator_138,
                                 matrix_accumulator_139);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_56,
                                 matrix_fragment_57);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_40,
                                 matrix_fragment_41);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_64, matrix_fragment_65, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_54,
                                 matrix_fragment_55);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_38,
                                 matrix_fragment_39);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_64, matrix_fragment_65, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_134,
                                 matrix_accumulator_135);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_66, matrix_fragment_67, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_66, matrix_fragment_67, matrix_accumulator_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_68, matrix_fragment_69, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_68, matrix_fragment_69, matrix_fragment_18,
                                 matrix_fragment_19);

        multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_130,
                                 matrix_accumulator_131);

        multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_122,
                                 matrix_accumulator_123);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_70, matrix_fragment_71, matrix_accumulator_20,
                                 matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_128,
                                 matrix_accumulator_129);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_72, matrix_fragment_73, matrix_accumulator_22,
                                 matrix_accumulator_23);

        multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_142,
                                 matrix_accumulator_143);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_74, matrix_fragment_75, matrix_accumulator_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_74, matrix_fragment_75, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_110,
                                 matrix_accumulator_111);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_50,
                                 matrix_fragment_51);

        multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_2,
                                 matrix_accumulator_3);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_48,
                                 matrix_fragment_49);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_4,
                                 matrix_fragment_5);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_80, matrix_fragment_81, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_84,
                                 matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_104,
                                 matrix_fragment_105);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_46,
                                 matrix_fragment_47);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_44,
                                 matrix_fragment_45);

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_92,
                                 matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                                 matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_80, matrix_fragment_81, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_96,
                                 matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                 matrix_fragment_82, matrix_fragment_83, matrix_accumulator_32,
                                 matrix_accumulator_33);
    } while (!uniform_condition_0);
    thread_row_156 = threadIdx.y;

    condition_5 = int32_t(packed_half_0) != int32_t(0u);

    condition_6 = int32_t(packed_half_144) != int32_t(0u);

    split_address_words(matrix_fragment_80, matrix_fragment_81,
                        uint64_t(uint32_t(thread_row_156)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_80 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_80, matrix_fragment_81) + 0x98220ull, memory_ranges,
        error_counts);

    address_word_163 = thread_row_156 * 0x800u;

    __syncthreads();

    split_address_words(address_word_162, address_word_163,
                        uint64_t(uint32_t(address_word_163)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    split_address_words(address_word_162, address_word_163,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_162, address_word_163));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                   select_half_operand<2, false, false>(packed_half_0));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88220ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88420ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88820ull, matrix_fragment_96,
        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_61 = result_word;
    }

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    condition_0 = (address_word_145 & 0x1u) != 0;

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    condition_1 = (address_word_145 & 0x2u) != 0;

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    condition_3 = (address_word_145 & 0x10u) != 0;

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_51 = transpose_half_tile(matrix_fragment_51);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                     select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_49 = transpose_half_tile(matrix_fragment_49);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_142));

    matrix_fragment_47 = transpose_half_tile(matrix_fragment_47);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_accumulator_143));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_46 = transpose_half_tile(matrix_fragment_46);

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_45 = transpose_half_tile(matrix_fragment_45);

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_accumulator_109 = transpose_half_tile(matrix_accumulator_109);

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_107 = transpose_half_tile(matrix_accumulator_107);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_105 = transpose_half_tile(matrix_fragment_105);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_125));

    matrix_accumulator_106 = transpose_half_tile(matrix_accumulator_106);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_104 = transpose_half_tile(matrix_fragment_104);

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_3 = transpose_half_tile(matrix_accumulator_3);

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_2 = transpose_half_tile(matrix_accumulator_2);

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_accumulator_113),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_accumulator_115),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_27 = transpose_half_tile(matrix_fragment_27);

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_26 = transpose_half_tile(matrix_fragment_26);

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_29 = transpose_half_tile(matrix_fragment_29);

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_31 = transpose_half_tile(matrix_fragment_31);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88620ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_144), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_accumulator_22));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_accumulator_23));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    matrix_fragment_90 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    matrix_fragment_88 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_65));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                   select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_5 = transpose_half_tile(matrix_fragment_5);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                     select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_87 = result_word;
    }

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_86 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_79), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_74 = result_word;
    }

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                     select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_7 = transpose_half_tile(matrix_fragment_7);

    matrix_fragment_81 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_83 = permute_signed(matrix_fragment_66, matrix_fragment_66, 0x5432u);

    matrix_fragment_89 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_9 = transpose_half_tile(matrix_fragment_9);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_83));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                   select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_64 = permute_signed(matrix_fragment_81, matrix_fragment_81, 0x5432u);

    matrix_accumulator_33 = transpose_half_tile(matrix_accumulator_33);

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x4100410u);

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    packed_half_160 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, address_word_161) + 0xa8240ull, memory_ranges, error_counts);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_82 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_65), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_152 = result_word;
    }

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x4100410u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_74 = permute_signed(matrix_fragment_75, matrix_fragment_75, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_78), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_82 = result_word;
    }

    matrix_fragment_76 = half_to_float_bits((matrix_fragment_88 & 65535u) * 65537u);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_88 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_154), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_89 = half_to_float_bits((matrix_fragment_66 >> 16) * 65537u);

    matrix_fragment_74 = half_to_float_bits((matrix_fragment_66 & 65535u) * 65537u);

    matrix_fragment_77 = rsqrt_bits(matrix_fragment_88);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x4100410u);

    matrix_fragment_66 = rsqrt_bits(matrix_fragment_74);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_75 & 65535u) * 65537u);

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_75 >> 16) * 65537u);

    matrix_fragment_89 = rsqrt_bits(matrix_fragment_89);

    matrix_fragment_75 = half_to_float_bits((matrix_fragment_88 & 65535u) * 65537u);

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_88 >> 16) * 65537u);

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                     select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_65 = permute_signed(matrix_fragment_64, matrix_fragment_64, 0x5432u);

    matrix_fragment_78 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_fragment_75 = rsqrt_bits(matrix_fragment_75);

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_82 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_88 = rsqrt_bits(matrix_fragment_88);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                     select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_83 = pack_half(matrix_fragment_89, matrix_fragment_66);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_144), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    matrix_fragment_74 = rsqrt_bits(matrix_fragment_90);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_81 = half_to_float_bits((matrix_fragment_72 & 65535u) * 65537u);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88c20ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_83 = half_to_float_bits((matrix_fragment_89 & 65535u) * 65537u);

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_89 >> 16) * 65537u);

    matrix_fragment_76 = rsqrt_bits(matrix_fragment_76);

    matrix_fragment_89 = pack_half(matrix_fragment_88, matrix_fragment_75);

    matrix_fragment_83 = rsqrt_bits(matrix_fragment_83);

    matrix_fragment_88 = rsqrt_bits(matrix_fragment_90);

    matrix_fragment_91 = pack_half(matrix_fragment_73, matrix_fragment_74);

    matrix_fragment_72 = half_to_float_bits((matrix_fragment_72 >> 16) * 65537u);

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_143 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_91 = pack_half(matrix_fragment_77, matrix_fragment_76);

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_82 = rsqrt_bits(matrix_fragment_72);

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                     select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_124 = pack_half(matrix_fragment_88, matrix_fragment_83);

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88a20ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x88e20ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89020ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_81 = rsqrt_bits(matrix_fragment_81);

    matrix_accumulator_120 = pack_half(0u, matrix_fragment_80);

    matrix_fragment_81 = pack_half(matrix_fragment_82, matrix_fragment_81);

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_81 = permute_signed(matrix_fragment_78, matrix_fragment_78, 0x5432u);

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_132 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                       select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_78 = permute_signed(matrix_fragment_79, matrix_fragment_79, 0x5432u);

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_79 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x4100410u);

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                                       select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_accumulator_138 = half_to_float_bits((matrix_accumulator_132 & 65535u) * 65537u);

    matrix_accumulator_134 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x4100410u);

    matrix_accumulator_135 = half_to_float_bits((matrix_accumulator_132 >> 16) * 65537u);

    matrix_accumulator_128 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_78 = rsqrt_bits(matrix_accumulator_138);

    matrix_accumulator_136 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_79 = rsqrt_bits(matrix_accumulator_135);

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_69 = half_to_float_bits((matrix_accumulator_134 & 65535u) * 65537u);

    matrix_accumulator_122 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 = half_to_float_bits((matrix_accumulator_134 >> 16) * 65537u);

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(matrix_accumulator_136));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89220ull, matrix_fragment_80,
        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83, memory_ranges, error_counts);

    matrix_fragment_69 = rsqrt_bits(matrix_fragment_69);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_accumulator_134 = rsqrt_bits(matrix_accumulator_134);

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_71 = permute_signed(packed_half_152, packed_half_152, 0x5432u);

    matrix_accumulator_135 = pack_half(matrix_fragment_79, matrix_fragment_78);

    packed_half_152 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                     select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_122 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                      select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_152), 0x4100410u);

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_138 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_71 = permute_signed(packed_half_154, packed_half_154, 0x5432u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89420ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_accumulator_112 = pack_half(matrix_accumulator_134, matrix_fragment_69);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_136 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 = half_to_float_bits((matrix_accumulator_114 & 65535u) * 65537u);

    matrix_accumulator_101 = half_to_float_bits((matrix_accumulator_114 >> 16) * 65537u);

    packed_half_154 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                     select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_accumulator_100 = rsqrt_bits(matrix_accumulator_100);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    packed_half_154 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_154), 0x4100410u);

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89620ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    matrix_accumulator_101 = rsqrt_bits(matrix_accumulator_101);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_accumulator_102 = half_to_float_bits((packed_half_154 & 65535u) * 65537u);

    packed_half_154 = half_to_float_bits((packed_half_154 >> 16) * 65537u);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_accumulator_114 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_118 = rsqrt_bits(matrix_accumulator_102);

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 = rsqrt_bits(packed_half_154);

    matrix_fragment_61 = permute_signed(packed_half_144, packed_half_144, 0x5432u);

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(matrix_fragment_60));

    packed_half_144 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                     select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_61 = pack_half(matrix_accumulator_101, matrix_accumulator_100);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89820ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_accumulator_102, matrix_accumulator_103, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_63 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x4100410u);

    matrix_fragment_62 = pack_half(matrix_accumulator_119, matrix_accumulator_118);

    matrix_accumulator_119 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                           select_half_operand<2, false, false>(0u));

    packed_half_144 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_144), 0x4100410u);

    packed_half_0 = half_to_float_bits((matrix_accumulator_20 & 65535u) * 65537u);

    matrix_accumulator_119 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_accumulator_121 = half_to_float_bits((matrix_accumulator_20 >> 16) * 65537u);

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_60 = half_to_float_bits((packed_half_144 & 65535u) * 65537u);

    matrix_fragment_25 = half_to_float_bits((packed_half_144 >> 16) * 65537u);

    packed_half_0 = rsqrt_bits(packed_half_0);

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_118 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_121 = rsqrt_bits(matrix_accumulator_121);

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_60 = rsqrt_bits(matrix_fragment_60);

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = rsqrt_bits(matrix_fragment_25);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_accumulator_121 = pack_half(matrix_accumulator_121, packed_half_0);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89a20ull, matrix_accumulator_20,
        matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
        error_counts);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_93 = pack_half(matrix_fragment_25, matrix_fragment_60);

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_43),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89c20ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_25), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_0 = result_word;
    }

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_92), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    packed_half_0 = permute_signed(matrix_fragment_57, matrix_fragment_57, 0x5432u);

    matrix_fragment_24 = permute_signed(matrix_fragment_93, matrix_fragment_93, 0x5432u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                   select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_93 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_0 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x4100410u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x4100410u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_25 = half_to_float_bits((packed_half_0 & 65535u) * 65537u);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_95 = half_to_float_bits((packed_half_0 >> 16) * 65537u);

    matrix_fragment_24 = half_to_float_bits((matrix_fragment_93 & 65535u) * 65537u);

    matrix_accumulator_121 = half_to_float_bits((matrix_fragment_93 >> 16) * 65537u);

    matrix_fragment_25 = rsqrt_bits(matrix_fragment_25);

    matrix_fragment_92 = rsqrt_bits(matrix_fragment_95);

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_24);

    matrix_accumulator_121 = rsqrt_bits(matrix_accumulator_121);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_92 = pack_half(matrix_fragment_92, matrix_fragment_25);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x89e20ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_121 = pack_half(matrix_accumulator_121, matrix_fragment_24);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_64,
                             matrix_fragment_65);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x8a020ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_fragment_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                      select_half_operand<2, false, false>(packed_half_10),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_94));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_25), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_24 = result_word;
    }

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_17));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_93 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_93));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                                   select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_25 = result_word;
    }

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_53));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_24 = permute_signed(matrix_fragment_25, matrix_fragment_25, 0x5432u);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x4100410u);

    matrix_fragment_53 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_24 = half_to_float_bits((matrix_fragment_25 & 65535u) * 65537u);

    matrix_fragment_25 = half_to_float_bits((matrix_fragment_25 >> 16) * 65537u);

    packed_half_0 = permute_signed(matrix_fragment_53, matrix_fragment_53, 0x5432u);

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_24);

    matrix_fragment_25 = rsqrt_bits(matrix_fragment_25);

    matrix_fragment_53 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                        select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x4100410u);

    packed_half_0 = half_to_float_bits((matrix_fragment_53 & 65535u) * 65537u);

    matrix_fragment_53 = half_to_float_bits((matrix_fragment_53 >> 16) * 65537u);

    matrix_fragment_24 = pack_half(matrix_fragment_25, matrix_fragment_24);

    packed_half_0 = rsqrt_bits(packed_half_0);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_53 = rsqrt_bits(matrix_fragment_53);

    packed_half_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                         select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    packed_half_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_accumulator_121));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_10 = 0x29c0u;

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_52 = pack_half(matrix_fragment_53, packed_half_0);

    matrix_accumulator_132 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_138 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x3c203c20u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_124 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3c203c20u);

    matrix_accumulator_133 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_134 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3c203c20u);

    matrix_accumulator_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_138 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138), 0x3c203c20u);

    matrix_accumulator_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_139 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_132 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x3e473e47u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_124 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_124), 0x3e473e47u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3e473e47u);

    matrix_accumulator_138 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_138), 0x3e473e47u);

    matrix_accumulator_133 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133), 0x3c203c20u);

    matrix_accumulator_125 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3c203c20u);

    matrix_accumulator_135 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3c203c20u);

    matrix_accumulator_139 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3c203c20u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_112),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_133 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133), 0x3e473e47u);

    matrix_accumulator_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_125 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_125), 0x3e473e47u);

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3e473e47u);

    matrix_accumulator_139 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_139), 0x3e473e47u);

    matrix_fragment_92 = (matrix_accumulator_132 << 0x5u) + 0x7ff88000u;

    matrix_fragment_55 = (matrix_accumulator_124 << 0x5u) + 0x7ff88000u;

    matrix_fragment_54 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    matrix_fragment_93 = (matrix_accumulator_138 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 = (matrix_accumulator_125 << 0x5u) + 0x7ff88000u;

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                   select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_52 = (matrix_accumulator_139 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_112 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_112), 0x3c203c20u);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_116 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3c203c20u);

    matrix_accumulator_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_113),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_82,
                             matrix_fragment_83);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_112 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_112), 0x3e473e47u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_116 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3e473e47u);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_113 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_113), 0x3c203c20u);

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_117 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3c203c20u);

    matrix_fragment_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_96),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_97),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_accumulator_113 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_113), 0x3e473e47u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<0, false, false>(matrix_accumulator_120));

    matrix_fragment_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_117 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3e473e47u);

    matrix_fragment_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_fragment_38 = (matrix_accumulator_133 << 0x5u) + 0x7ff88000u;

    matrix_fragment_39 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                   select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 = (matrix_accumulator_116 << 0x5u) + 0x7ff88000u;

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_11 = (matrix_accumulator_112 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_37 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_accumulator_16));

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_96 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96), 0x3e473e47u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97), 0x3e473e47u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_16 = (matrix_accumulator_113 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_25 = (matrix_accumulator_117 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    packed_half_0 = (matrix_fragment_96 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_accumulator_17 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_96 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_24 = (matrix_fragment_97 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_94 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_fragment_95 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_85 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_94 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_95 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_96 = (matrix_fragment_85 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_97 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_87 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_84 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_85 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                        select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                        select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_67 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_66 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_fragment_65 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_74 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_72 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_75 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_73 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_88 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_89 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_83),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_121 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_90 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_89 = condition_0 ? matrix_fragment_90 : matrix_accumulator_121;

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_80),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_121 = condition_0 ? matrix_accumulator_121 : matrix_fragment_90;

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    matrix_fragment_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_81),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_20,
                             matrix_accumulator_21);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3c203c20u);

    matrix_fragment_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_82),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_128, matrix_accumulator_129, matrix_accumulator_22,
                             matrix_accumulator_23);

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3c203c20u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3c203c20u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_80 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_81 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81), 0x3e473e47u);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_88 = condition_0 ? matrix_fragment_99 : matrix_accumulator_116;

    matrix_fragment_83 = (matrix_fragment_80 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_116 = condition_0 ? matrix_accumulator_116 : matrix_fragment_99;

    matrix_fragment_82 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_80 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_76 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_77 = (matrix_fragment_81 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_79 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_82 = (matrix_fragment_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_78 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_81 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_70 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_68 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_102 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3c203c20u);

    matrix_accumulator_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_accumulator_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3c203c20u);

    matrix_accumulator_22 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_22),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                        select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_102 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3e473e47u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_98 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_accumulator_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_69 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_accumulator_103 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3e473e47u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_20 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3c203c20u);

    matrix_accumulator_21 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3c203c20u);

    matrix_accumulator_22 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_22), 0x3c203c20u);

    matrix_accumulator_23 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23), 0x3c203c20u);

    matrix_accumulator_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    packed_half_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<0, false, false>(packed_half_10), 0x3d343d34u);

    matrix_fragment_62 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_113 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_61 = (matrix_accumulator_102 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_20 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3e473e47u);

    matrix_accumulator_21 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3e473e47u);

    matrix_accumulator_22 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_22), 0x3e473e47u);

    matrix_accumulator_23 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23), 0x3e473e47u);

    matrix_fragment_60 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_fragment_63 = (matrix_accumulator_103 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_accumulator_113 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_accumulator_101 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_118 = !condition_1 ? matrix_fragment_88 : matrix_fragment_89;

    matrix_accumulator_119 = !condition_1 ? matrix_fragment_89 : matrix_fragment_88;

    matrix_accumulator_100 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    packed_half_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0x3c203c20u);

    matrix_fragment_89 = (matrix_accumulator_20 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = (matrix_accumulator_21 << 0x5u) + 0x7ff88000u;

    matrix_fragment_88 = (matrix_accumulator_22 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = (matrix_accumulator_23 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3c203c20u);

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0x3e473e47u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    matrix_fragment_18 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                        select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3e473e47u);

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_accumulator_113 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    packed_half_10 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_117 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_114 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_103 = (matrix_fragment_42 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                       select_half_operand<2, false, false>(packed_half_10));

    matrix_fragment_14 = address_word_145 * 0x4u;

    matrix_fragment_18 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_102 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_115 = (matrix_accumulator_115 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 = matrix_fragment_14 & 0x1cu;

    matrix_fragment_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                       select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_19 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_15 =
        uint32_t(int64_t(int32_t(address_word_145)) * (1ll << 0x1du) >> 32) + matrix_fragment_18;

    matrix_accumulator_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_14 = !condition_1 ? matrix_accumulator_116 : matrix_accumulator_121;

    matrix_accumulator_116 = !condition_1 ? matrix_accumulator_121 : matrix_accumulator_116;

    matrix_fragment_18 = condition_0 ? matrix_fragment_13 : matrix_fragment_12;

    matrix_accumulator_22 = condition_0 ? matrix_fragment_12 : matrix_fragment_13;

    matrix_fragment_13 = 0x1u ^ matrix_fragment_15;

    matrix_fragment_12 = 0x2u ^ matrix_fragment_15;

    matrix_accumulator_21 = 0x3u ^ matrix_fragment_15;

    matrix_fragment_41 = condition_0 ? matrix_fragment_19 : matrix_accumulator_20;

    matrix_accumulator_23 = condition_0 ? matrix_accumulator_20 : matrix_fragment_19;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_118), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_118 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    matrix_fragment_40 = !condition_1 ? matrix_accumulator_22 : matrix_fragment_41;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_119), "r"(matrix_fragment_12), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_119 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_accumulator_21), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_116 = result_word;
    }

    matrix_accumulator_20 = !condition_1 ? matrix_fragment_18 : matrix_accumulator_23;

    matrix_fragment_43 = !condition_1 ? matrix_accumulator_23 : matrix_fragment_18;

    matrix_accumulator_22 = !condition_1 ? matrix_fragment_41 : matrix_accumulator_22;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_20), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_20 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_43), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_22), "r"(matrix_accumulator_21), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_22 = result_word;
    }

    condition_0 = (address_word_145 & 0x8u) != 0;

    matrix_fragment_14 = !condition_0 ? matrix_accumulator_118 : matrix_fragment_19;

    matrix_accumulator_23 = !condition_0 ? matrix_accumulator_119 : matrix_accumulator_116;

    matrix_fragment_18 = !condition_3 ? matrix_fragment_14 : matrix_accumulator_23;

    matrix_fragment_14 = !condition_3 ? matrix_accumulator_23 : matrix_fragment_14;

    matrix_accumulator_23 = uint32_t(int32_t(address_word_147) >> 0x2u);

    matrix_fragment_15 = !condition_0 ? matrix_accumulator_20 : matrix_fragment_41;

    matrix_fragment_19 = !condition_0 ? matrix_fragment_19 : matrix_accumulator_118;

    matrix_accumulator_116 = !condition_0 ? matrix_accumulator_116 : matrix_accumulator_119;

    matrix_accumulator_20 = !condition_0 ? matrix_fragment_41 : matrix_accumulator_20;

    matrix_fragment_41 = matrix_accumulator_23 + 0x8u;

    matrix_fragment_12 = !condition_0 ? matrix_fragment_43 : matrix_accumulator_22;

    matrix_fragment_43 = !condition_0 ? matrix_accumulator_22 : matrix_fragment_43;

    matrix_fragment_13 = !condition_3 ? matrix_fragment_19 : matrix_accumulator_116;

    matrix_fragment_42 = uint32_t(int32_t(matrix_fragment_41) >> 0x1fu);

    matrix_accumulator_21 = !condition_3 ? matrix_fragment_15 : matrix_fragment_12;

    matrix_accumulator_22 = !condition_3 ? matrix_accumulator_20 : matrix_fragment_43;

    matrix_fragment_12 = !condition_3 ? matrix_fragment_12 : matrix_fragment_15;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_42 =
        uint32_t(combine_words(matrix_fragment_42, 0u) << 0x6u >> 32) + matrix_fragment_41;

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                       select_half_operand<2, false, false>(matrix_accumulator_22));

    matrix_fragment_19 = !condition_3 ? matrix_accumulator_116 : matrix_fragment_19;

    matrix_accumulator_20 = !condition_3 ? matrix_fragment_43 : matrix_accumulator_20;

    matrix_fragment_40 = uint32_t(int32_t(address_word_147) >> 0x1fu);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_42 = matrix_fragment_42 & 0xffc0u;

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_40 =
        uint32_t(combine_words(matrix_fragment_40, 0u) << 0x6u >> 32) + matrix_accumulator_23;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_42 = matrix_fragment_41 + (0u - matrix_fragment_42);

    matrix_fragment_40 = matrix_fragment_40 & 0xffc0u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_13),
                                        select_half_operand<1, false, false>(matrix_fragment_13));

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_21),
                       select_half_operand<1, false, false>(matrix_accumulator_21));

    matrix_fragment_15 = permute_signed(matrix_fragment_42, 0u, 0x8880u);

    matrix_fragment_40 = matrix_accumulator_23 + (0u - matrix_fragment_40);

    matrix_fragment_12 = permute_signed(matrix_fragment_13, matrix_fragment_12, 0x5410u);

    matrix_fragment_13 = permute_signed(matrix_fragment_15, 0u, 0x7710u);

    matrix_fragment_14 = permute_signed(matrix_fragment_40, 0u, 0x8880u);

    matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_13) >> 0xau);

    matrix_fragment_14 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_fragment_19 = matrix_fragment_18 & 0x1fu;

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_14) >> 0xau);

    matrix_fragment_19 = matrix_fragment_42 + matrix_fragment_19;

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    matrix_fragment_15 = matrix_fragment_14 & 0x1fu;

    matrix_fragment_18 = matrix_fragment_19 & 0xe0u;

    matrix_fragment_13 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    matrix_accumulator_22 =
        uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_fragment_15 = matrix_fragment_40 + matrix_fragment_15;

    matrix_fragment_18 = 0u - matrix_fragment_18;

    matrix_accumulator_20 =
        uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_accumulator_23 = uint32_t(int32_t(matrix_accumulator_22) >> 0x1fu);

    matrix_fragment_14 = matrix_fragment_15 & 0xe0u;

    matrix_fragment_13 = reciprocal_bits(matrix_fragment_13);

    matrix_fragment_43 = permute_signed(matrix_fragment_18, 0u, 0x7710u);

    matrix_accumulator_21 = uint32_t(int32_t(matrix_accumulator_20) >> 0x1fu);

    matrix_accumulator_23 =
        uint32_t(combine_words(matrix_accumulator_23, 0u) << 0x6u >> 32) + matrix_accumulator_22;

    matrix_fragment_12 = reciprocal_bits(matrix_fragment_12);

    matrix_fragment_14 = 0u - matrix_fragment_14;

    matrix_accumulator_21 =
        uint32_t(combine_words(matrix_accumulator_21, 0u) << 0x6u >> 32) + matrix_accumulator_20;

    matrix_fragment_42 = matrix_fragment_42 + matrix_fragment_43;

    matrix_accumulator_23 = matrix_accumulator_23 & 0xffc0u;

    matrix_fragment_41 = permute_signed(matrix_fragment_14, 0u, 0x7710u);

    matrix_accumulator_21 = matrix_accumulator_21 & 0xffc0u;

    matrix_fragment_42 = matrix_fragment_42 & 0xffffu;

    matrix_accumulator_23 = matrix_accumulator_22 + (0u - matrix_accumulator_23);

    matrix_fragment_40 = matrix_fragment_40 + matrix_fragment_41;

    matrix_accumulator_21 = matrix_accumulator_20 + (0u - matrix_accumulator_21);

    matrix_fragment_42 = permute_signed(matrix_fragment_42, 0u, 0x8880u);

    matrix_fragment_18 = permute_signed(matrix_accumulator_23, 0u, 0x8880u);

    matrix_accumulator_116 = pack_half(matrix_fragment_12, matrix_fragment_13);

    matrix_fragment_40 = matrix_fragment_40 & 0xffffu;

    matrix_accumulator_20 = matrix_fragment_15 & 0xffffu;

    matrix_fragment_12 = permute_signed(matrix_accumulator_21, 0u, 0x8880u);

    matrix_fragment_15 = matrix_fragment_42;

    matrix_fragment_18 = permute_signed(matrix_fragment_18, 0u, 0x7710u);

    matrix_fragment_13 = permute_signed(matrix_fragment_40, 0u, 0x8880u);

    matrix_fragment_12 = permute_signed(matrix_fragment_12, 0u, 0x7710u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_18) >> 0xau);

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_12) >> 0xau);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_13), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_12 = uint32_t(uint32_t(matrix_accumulator_20) >> 0x5u);

    matrix_fragment_18 = matrix_fragment_18 & 0x1fu;

    matrix_fragment_14 = matrix_fragment_14 & 0x1fu;

    condition_0 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_fragment_43 = matrix_accumulator_23 + matrix_fragment_18;

    matrix_fragment_12 = transpose_half_tile(matrix_accumulator_111);

    matrix_fragment_19 = matrix_fragment_19 & 0xffffu;

    matrix_fragment_42 = matrix_accumulator_21 + matrix_fragment_14;

    matrix_accumulator_20 = matrix_fragment_43 & 0xe0u;

    matrix_fragment_18 = transpose_half_tile(matrix_accumulator_110);

    matrix_fragment_14 = uint32_t(uint32_t(matrix_fragment_19) >> 0x5u);

    matrix_fragment_19 = matrix_fragment_42 & 0xe0u;

    matrix_accumulator_20 = 0u - matrix_accumulator_20;

    matrix_fragment_56 =
        uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1eu) >> 32) + 0x20u;

    matrix_accumulator_22 = 0u - matrix_fragment_19;

    condition_1 = (matrix_fragment_14 & 0x1u) != 0;

    matrix_fragment_40 = permute_signed(matrix_accumulator_20, 0u, 0x7710u);

    matrix_accumulator_20 = permute_signed(matrix_fragment_15, matrix_accumulator_20, 0x7632u);

    matrix_fragment_41 = uint32_t(int32_t(matrix_fragment_56) >> 0x1fu);

    matrix_fragment_14 = permute_signed(matrix_fragment_13, matrix_fragment_14, 0x7632u);

    matrix_accumulator_22 = permute_signed(matrix_accumulator_22, 0u, 0x7710u);

    matrix_accumulator_110 = condition_1 ? matrix_accumulator_20 : matrix_fragment_15;

    matrix_fragment_41 =
        uint32_t(combine_words(matrix_fragment_41, 0u) << 0x6u >> 32) + matrix_fragment_56;

    matrix_fragment_40 = matrix_accumulator_23 + matrix_fragment_40;

    matrix_accumulator_111 = condition_0 ? matrix_fragment_14 : matrix_fragment_13;

    matrix_accumulator_21 = matrix_accumulator_21 + matrix_accumulator_22;

    matrix_accumulator_23 = transpose_half_tile(matrix_fragment_48);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_41 = matrix_fragment_41 & 0xffc0u;

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_48 =
        uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1eu) >> 32) + 0x28u;

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_21 = matrix_accumulator_21 & 0xffffu;

    matrix_fragment_39 = uint32_t(int32_t(matrix_fragment_48) >> 0x1fu);

    matrix_fragment_56 = matrix_fragment_56 + (0u - matrix_fragment_41);

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_39 =
        uint32_t(combine_words(matrix_fragment_39, 0u) << 0x6u >> 32) + matrix_fragment_48;

    matrix_accumulator_21 = permute_signed(matrix_accumulator_21, 0u, 0x8880u);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_accumulator_20 = permute_signed(matrix_fragment_56, 0u, 0x8880u);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = matrix_fragment_39 & 0xffc0u;

    matrix_fragment_40 = matrix_fragment_40 & 0xffffu;

    matrix_fragment_39 = matrix_accumulator_21;

    matrix_accumulator_20 = permute_signed(matrix_accumulator_20, 0u, 0x7710u);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_51 = permute_signed(matrix_fragment_40, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_39), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_fragment_40 = uint32_t(uint32_t(matrix_accumulator_20) >> 0xau);

    matrix_fragment_19 = transpose_half_tile(matrix_fragment_50);

    matrix_fragment_41 = matrix_fragment_40 & 0x1fu;

    matrix_fragment_50 = transpose_half_tile(matrix_fragment_44);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = transpose_half_tile(matrix_accumulator_108);

    matrix_fragment_49 = matrix_fragment_48 + (0u - matrix_fragment_53);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_48 = matrix_fragment_56 + matrix_fragment_41;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_51), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    matrix_fragment_52 = matrix_fragment_43 & 0xffffu;

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_38 = matrix_fragment_48 & 0xe0u;

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_39 = matrix_fragment_42 & 0xffffu;

    matrix_fragment_46 = permute_signed(matrix_fragment_49, 0u, 0x8880u);

    matrix_fragment_53 = 0u - matrix_fragment_38;

    matrix_fragment_38 = uint32_t(uint32_t(matrix_fragment_39) >> 0x5u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_fragment_46 = permute_signed(matrix_fragment_46, 0u, 0x7710u);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    condition_0 = (matrix_fragment_38 & 0x1u) != 0;

    matrix_fragment_44 = permute_signed(matrix_fragment_93, matrix_fragment_44, 0x7632u);

    matrix_fragment_46 = uint32_t(uint32_t(matrix_fragment_46) >> 0xau);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_50 = matrix_fragment_46 & 0x1fu;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_93 = condition_0 ? matrix_fragment_44 : matrix_fragment_93;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                      select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_42 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_39 = uint32_t(uint32_t(matrix_fragment_52) >> 0x5u);

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                       select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_47 = permute_signed(matrix_fragment_53, 0u, 0x7710u);

    matrix_fragment_54 = matrix_fragment_49 + matrix_fragment_50;

    condition_1 = (matrix_fragment_39 & 0x1u) != 0;

    matrix_fragment_92 = permute_signed(matrix_fragment_51, matrix_fragment_92, 0x7632u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_56 = matrix_fragment_56 + matrix_fragment_47;

    matrix_fragment_92 = condition_1 ? matrix_fragment_92 : matrix_fragment_51;

    matrix_fragment_51 =
        uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1eu) >> 32) + 0x30u;

    matrix_fragment_56 = matrix_fragment_56 & 0xffffu;

    matrix_fragment_50 = uint32_t(int32_t(matrix_fragment_51) >> 0x1fu);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_50 =
        uint32_t(combine_words(matrix_fragment_50, 0u) << 0x6u >> 32) + matrix_fragment_51;

    matrix_fragment_53 = permute_signed(matrix_fragment_56, 0u, 0x8880u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    address_word_147 = uint32_t(int64_t(int32_t(address_word_147)) * (1ll << 0x1eu) >> 32) + 0x38u;

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_14 = matrix_fragment_54 & 0xe0u;

    matrix_fragment_52 = 0u - matrix_fragment_14;

    matrix_fragment_56 = matrix_fragment_50 & 0xffc0u;

    matrix_fragment_52 = permute_signed(matrix_fragment_52, 0u, 0x7710u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = matrix_fragment_49 + matrix_fragment_52;

    matrix_fragment_49 = matrix_fragment_53;

    matrix_fragment_58 = uint32_t(int32_t(address_word_147) >> 0x1fu);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_58 =
        uint32_t(combine_words(matrix_fragment_58, 0u) << 0x6u >> 32) + address_word_147;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_49), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_84 = result_word;
    }

    matrix_fragment_56 = matrix_fragment_51 + (0u - matrix_fragment_56);

    matrix_fragment_50 = matrix_fragment_50 & 0xffffu;

    matrix_fragment_58 = matrix_fragment_58 & 0xffc0u;

    matrix_fragment_57 = permute_signed(matrix_fragment_50, 0u, 0x8880u);

    matrix_fragment_51 = permute_signed(matrix_fragment_56, 0u, 0x8880u);

    address_word_147 = address_word_147 + (0u - matrix_fragment_58);

    matrix_fragment_50 = permute_signed(matrix_fragment_51, 0u, 0x7710u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_57), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_49 = permute_signed(address_word_147, 0u, 0x8880u);

    matrix_fragment_55 = uint32_t(uint32_t(matrix_fragment_50) >> 0xau);

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_48 = matrix_fragment_48 & 0xffffu;

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = matrix_fragment_55 & 0x1fu;

    matrix_fragment_49 = permute_signed(matrix_fragment_49, 0u, 0x7710u);

    matrix_fragment_48 = uint32_t(uint32_t(matrix_fragment_48) >> 0x5u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_59 = uint32_t(uint32_t(matrix_fragment_49) >> 0xau);

    matrix_fragment_85 = matrix_fragment_56 + matrix_fragment_55;

    condition_0 = (matrix_fragment_48 & 0x1u) != 0;

    matrix_fragment_55 = permute_signed(matrix_fragment_84, matrix_fragment_55, 0x7632u);

    matrix_fragment_54 = matrix_fragment_54 & 0xffffu;

    matrix_fragment_86 = matrix_fragment_59 & 0x1fu;

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_58 = matrix_fragment_85 & 0xe0u;

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_84 = condition_0 ? matrix_fragment_55 : matrix_fragment_84;

    matrix_fragment_54 = uint32_t(uint32_t(matrix_fragment_54) >> 0x5u);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_86 = address_word_147 + matrix_fragment_86;

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = 0u - matrix_fragment_58;

    condition_0 = (matrix_fragment_54 & 0x1u) != 0;

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_82 = permute_signed(matrix_fragment_57, matrix_fragment_82, 0x7632u);

    matrix_fragment_58 = matrix_fragment_86 & 0xe0u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    matrix_fragment_55 = permute_signed(matrix_fragment_55, 0u, 0x7710u);

    matrix_fragment_82 = condition_0 ? matrix_fragment_82 : matrix_fragment_57;

    matrix_fragment_57 = 0u - matrix_fragment_58;

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_56 = matrix_fragment_56 + matrix_fragment_55;

    matrix_fragment_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_76 = permute_signed(matrix_fragment_57, 0u, 0x7710u);

    address_word_147 = address_word_147 + matrix_fragment_76;

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = matrix_fragment_56 & 0xffffu;

    address_word_147 = address_word_147 & 0xffffu;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(matrix_fragment_94));

    matrix_fragment_79 = permute_signed(matrix_fragment_58, 0u, 0x8880u);

    matrix_fragment_83 = permute_signed(address_word_147, 0u, 0x8880u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_79), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_79 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(matrix_fragment_83), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_83 = result_word;
    }

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_78 = matrix_fragment_85 & 0xffffu;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_80 = matrix_fragment_86 & 0xffffu;

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_78 = uint32_t(uint32_t(matrix_fragment_78) >> 0x5u);

    matrix_fragment_80 = uint32_t(uint32_t(matrix_fragment_80) >> 0x5u);

    condition_0 = (matrix_fragment_78 & 0x1u) != 0;

    condition_1 = (matrix_fragment_80 & 0x1u) != 0;

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_78 = permute_signed(matrix_fragment_79, matrix_fragment_78, 0x7632u);

    matrix_fragment_80 = permute_signed(matrix_fragment_83, matrix_fragment_80, 0x7632u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_81 = condition_0 ? matrix_fragment_78 : matrix_fragment_79;

    matrix_fragment_80 = condition_1 ? matrix_fragment_80 : matrix_fragment_83;

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_15 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                      select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_25 = transpose_half_tile(matrix_fragment_28);

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<0, false, false>(matrix_accumulator_111));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<0, false, false>(matrix_accumulator_110));

    matrix_fragment_27 = transpose_half_tile(matrix_fragment_30);

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_24 = transpose_half_tile(matrix_fragment_4);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_fragment_26 = transpose_half_tile(matrix_fragment_6);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_16 = transpose_half_tile(matrix_fragment_8);

    matrix_accumulator_17 = transpose_half_tile(matrix_accumulator_32);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<0, false, false>(matrix_fragment_93));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_accumulator_16 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<0, false, false>(matrix_fragment_92));

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<0, false, false>(matrix_fragment_84));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<0, false, false>(matrix_fragment_81));

    packed_half_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                         select_half_operand<0, false, false>(matrix_fragment_80));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<0, false, false>(matrix_fragment_82));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_56,
                             matrix_fragment_57);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_58,
                             matrix_fragment_59);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<0, false, false>(matrix_fragment_80));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                                      select_half_operand<2, false, false>(matrix_accumulator_117));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(packed_half_10));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = uint32_t(uint32_t(index_word_148) >> 0x1u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_44,
                             matrix_fragment_45);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_accumulator_3 = packed_half_0 + 0x8u;

    matrix_accumulator_2 = packed_half_0 + 0x4u;

    packed_half_0 = packed_half_0 + 0xcu;

    matrix_accumulator_3 = matrix_accumulator_3 + address_word_145 + (0u - address_word_146);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_77 = matrix_accumulator_2 + address_word_145 + (0u - address_word_146);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        uint64_t(uint32_t(matrix_accumulator_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_11 = packed_half_0 + address_word_145 + (0u - address_word_146);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_33 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(packed_half_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_32 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_10, matrix_fragment_11) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                         select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                         select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_76, matrix_fragment_77,
                        uint64_t(uint32_t(matrix_fragment_77)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                  select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x2000u, matrix_fragment_48,
                                        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_76 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_76, matrix_fragment_77) + 0xa8240ull, memory_ranges,
        error_counts);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x3000u, matrix_fragment_56,
                                        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                         select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    load_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149, matrix_accumulator_16, matrix_accumulator_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    load_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x1000u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, tile_word_149, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x1000u, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x2000u, matrix_fragment_28,
                                         matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149 + 0x3000u, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_81 = thread_row_156 * 0x100u;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_81)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x98240ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x98440ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_9 = matrix_fragment_81 + 0x800u;

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x98440ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x98240ull, matrix_fragment_8,
        matrix_fragment_9, packed_half_10, matrix_fragment_11, memory_ranges, error_counts);

    matrix_fragment_15 = matrix_fragment_81 + 0x1000u;

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_14, matrix_fragment_15));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x98240ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x98440ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    packed_half_0 = (address_word_145 << 0x4u) + uniform_index_word_6;

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0, matrix_fragment_104, matrix_fragment_105,
        matrix_accumulator_106, matrix_accumulator_107, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_17);

    matrix_accumulator_3 = matrix_fragment_81 + 0x1800u;

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_18);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2000u, matrix_fragment_64,
                                        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_160));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_19);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1000u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_80 = unpack_fp8_bits(matrix_accumulator_16);

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    matrix_accumulator_16 = unpack_fp8_bits(matrix_accumulator_16 >> 16);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3000u, matrix_fragment_68,
                                        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x98240ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_40);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x200u, matrix_fragment_72,
                                        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_48);

    matrix_fragment_48 = unpack_fp8_bits(matrix_fragment_48 >> 16);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_40 >> 16);

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_56);

    matrix_fragment_56 = unpack_fp8_bits(matrix_fragment_56 >> 16);

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x98440ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_41);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_3 = matrix_fragment_81 + 0x2000u;

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x98440ull,
        matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23,
        memory_ranges, error_counts);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_41 >> 16);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_16,
                             matrix_accumulator_17);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_62, matrix_fragment_63, matrix_accumulator_16,
                             matrix_accumulator_17);

    matrix_accumulator_16 = unpack_fp8_bits(matrix_fragment_42);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_fragment_43);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_16),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x98240ull,
        matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_18, matrix_fragment_19,
        memory_ranges, error_counts);

    matrix_fragment_42 = unpack_fp8_bits(matrix_fragment_42 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_49);

    matrix_fragment_49 = unpack_fp8_bits(matrix_fragment_49 >> 16);

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_3 = matrix_fragment_81 + 0x2800u;

    matrix_fragment_43 = unpack_fp8_bits(matrix_fragment_43 >> 16);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                             matrix_fragment_49);

    split_address_words(matrix_fragment_48, matrix_fragment_49,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_42,
                             matrix_fragment_43);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_48, matrix_fragment_49) + 0x98240ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_48, matrix_fragment_49) + 0x98440ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_50);

    matrix_fragment_49 = unpack_fp8_bits(matrix_fragment_57);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_51);

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_49 = matrix_fragment_81 + 0x3000u;

    matrix_fragment_50 = unpack_fp8_bits(matrix_fragment_50 >> 16);

    matrix_fragment_51 = unpack_fp8_bits(matrix_fragment_51 >> 16);

    matrix_fragment_57 = unpack_fp8_bits(matrix_fragment_57 >> 16);

    split_address_words(matrix_fragment_48, matrix_fragment_49,
                        int64_t(int32_t(matrix_fragment_49)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_76));

    split_address_words(matrix_fragment_56, matrix_fragment_57,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_48, matrix_fragment_49));

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_86,
                             matrix_fragment_87);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1200u, matrix_fragment_76,
                                        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_50,
                             matrix_fragment_51);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_56, matrix_fragment_57) + 0x98240ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_56, matrix_fragment_57) + 0x98440ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_59);

    matrix_fragment_59 = unpack_fp8_bits(matrix_fragment_59 >> 16);

    matrix_fragment_57 = unpack_fp8_bits(matrix_fragment_58);

    matrix_fragment_58 = unpack_fp8_bits(matrix_fragment_58 >> 16);

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_32));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_60, matrix_fragment_61, matrix_fragment_96,
                             matrix_fragment_97);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_33));

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_62, matrix_fragment_63, matrix_fragment_58,
                             matrix_fragment_59);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3200u, matrix_fragment_56,
                                        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_69 = matrix_fragment_81 + 0x3800u;

    split_address_words(matrix_fragment_68, matrix_fragment_69,
                        int64_t(int32_t(matrix_fragment_69)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    split_address_words(matrix_fragment_80, matrix_fragment_81,
                        int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_68, matrix_fragment_69));

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_90,
                             matrix_fragment_91);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_80, matrix_fragment_81) + 0x98240ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             packed_half_10, matrix_fragment_11, matrix_fragment_104,
                             matrix_fragment_105);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_108,
                             matrix_accumulator_109);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_80, matrix_fragment_81) + 0x98440ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_60, matrix_fragment_61, matrix_accumulator_2,
                             matrix_accumulator_3);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2200u, matrix_fragment_64,
                                        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x400u, matrix_fragment_60,
                                        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             packed_half_10, matrix_fragment_11, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_106,
                             matrix_accumulator_107);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1400u, matrix_fragment_72,
                                        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             packed_half_10, matrix_fragment_11, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_10, matrix_fragment_11, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2400u, matrix_fragment_8,
                                        matrix_fragment_9, packed_half_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3400u, matrix_fragment_56,
                                        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1600u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_106,
                             matrix_accumulator_107);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x600u, matrix_fragment_60,
                                        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2600u, matrix_fragment_8,
                                        matrix_fragment_9, packed_half_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3600u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x800u, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1800u, matrix_fragment_56,
                                        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_106,
                             matrix_accumulator_107);

    condition_2 = (address_word_145 & 0x4u) != 0;

    matrix_fragment_60 = parameters[parameter_output_low_2];
    matrix_fragment_61 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_108,
                             matrix_accumulator_109);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2800u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3800u, matrix_fragment_24,
                                        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2a00u, matrix_fragment_8,
                                        matrix_fragment_9, packed_half_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xa00u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1a00u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_22, matrix_accumulator_23, matrix_accumulator_106,
                             matrix_accumulator_107);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2e00u, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_22, matrix_accumulator_23, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_22, matrix_accumulator_23, matrix_accumulator_32,
                             matrix_accumulator_33);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0 + 0x3a00u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_82,
                             matrix_fragment_83);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x2c00u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_0 + 0xc00u, matrix_accumulator_16, matrix_accumulator_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_106,
                             matrix_accumulator_107);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1c00u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_108,
                             matrix_accumulator_109);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0xe00u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_92,
                             matrix_fragment_93);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_accumulator_32, matrix_accumulator_33, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_32,
                             matrix_accumulator_33);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_98,
                             matrix_fragment_99);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_accumulator_16,
                             matrix_accumulator_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x1e00u, matrix_fragment_8,
                                        matrix_fragment_9, packed_half_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_108,
                             matrix_accumulator_109);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3c00u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_84,
                             matrix_fragment_85);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_0 + 0x3e00u, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_92,
                             matrix_fragment_93);

    matrix_fragment_56 = blockIdx.y;

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_32,
                             matrix_accumulator_33);

    packed_half_0 = index_word_155 + 0x1u;

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_26,
                             matrix_fragment_27);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_accumulator_32 = index_word_155 * index_word_150 + index_word_150;

    matrix_accumulator_33 = address_word_145 & 0x13u;

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_24,
                             matrix_fragment_25);

    condition_0 = int32_t(packed_half_0) >= int32_t(index_word_151);

    index_word_150 = index_word_155 * index_word_150 + index_word_153;

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_106,
                             matrix_accumulator_107);

    packed_half_0 = index_word_153 + matrix_accumulator_32;

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_22,
                             matrix_accumulator_23);

    matrix_fragment_30 = uint32_t(uint32_t(address_word_145) >> 0x1u);

    matrix_fragment_31 = thread_row_156 * 0x80u;

    matrix_accumulator_32 = address_word_145 * 0x2u;

    matrix_fragment_63 = index_word_150 * 0x400u + matrix_fragment_31;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, matrix_fragment_11,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_94,
                             matrix_fragment_95);

    matrix_fragment_31 = packed_half_0 * 0x400u + matrix_fragment_31;

    matrix_fragment_41 = !condition_2 ? matrix_fragment_27 : matrix_fragment_19;

    multiply_accumulate_half(packed_half_10, matrix_fragment_11, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_76, matrix_fragment_77, matrix_accumulator_16,
                             matrix_accumulator_17);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_53 = parameters[parameter_scalar_control_11];

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_92,
                             matrix_fragment_93);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_68, matrix_fragment_69, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_37 = matrix_accumulator_33 | 0x4u & matrix_fragment_30;

    matrix_fragment_38 = !condition_2 ? matrix_fragment_26 : matrix_fragment_18;

    packed_half_0 = matrix_fragment_37 | 0x8u & matrix_accumulator_32;

    matrix_fragment_37 = !condition_2 ? matrix_fragment_12 : matrix_accumulator_20;

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_68, matrix_fragment_69, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_32 = !condition_2 ? matrix_fragment_13 : matrix_accumulator_21;

    matrix_fragment_45 = !condition_3 ? matrix_fragment_38 : matrix_fragment_41;

    matrix_fragment_38 = !condition_3 ? matrix_fragment_41 : matrix_fragment_38;

    matrix_fragment_41 = !condition_2 ? matrix_accumulator_22 : matrix_fragment_28;

    matrix_fragment_44 = !condition_3 ? matrix_fragment_37 : matrix_accumulator_32;

    matrix_fragment_37 = !condition_3 ? matrix_accumulator_32 : matrix_fragment_37;

    matrix_accumulator_32 = !condition_2 ? matrix_accumulator_23 : matrix_fragment_29;

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_50 = !condition_3 ? matrix_fragment_41 : matrix_accumulator_32;

    matrix_fragment_41 = !condition_3 ? matrix_accumulator_32 : matrix_fragment_41;

    condition_1 = int32_t(index_word_155) >= int32_t(index_word_151);

    matrix_accumulator_32 = matrix_fragment_56 * 0x8u + matrix_fragment_53;

    matrix_fragment_36 = !condition_2 ? matrix_fragment_18 : matrix_fragment_26;

    matrix_fragment_43 = !condition_2 ? matrix_fragment_19 : matrix_fragment_27;

    matrix_fragment_40 = !condition_2 ? matrix_fragment_28 : matrix_accumulator_22;

    matrix_fragment_51 = !condition_2 ? matrix_fragment_29 : matrix_accumulator_23;

    matrix_fragment_52 = !condition_2 ? matrix_fragment_70 : matrix_fragment_8;

    matrix_fragment_5 = !condition_2 ? matrix_fragment_71 : matrix_fragment_9;

    condition_1 = int32_t(matrix_accumulator_32) > int32_t(0u - 0x4u) && !condition_1;

    matrix_fragment_46 = !condition_3 ? matrix_fragment_36 : matrix_fragment_43;

    matrix_fragment_36 = !condition_3 ? matrix_fragment_43 : matrix_fragment_36;

    matrix_accumulator_33 = !condition_2 ? matrix_accumulator_20 : matrix_fragment_12;

    matrix_fragment_30 = !condition_2 ? matrix_accumulator_21 : matrix_fragment_13;

    matrix_fragment_42 = !condition_2 ? matrix_fragment_14 : matrix_fragment_24;

    matrix_fragment_43 = !condition_2 ? matrix_fragment_15 : matrix_fragment_25;

    matrix_fragment_49 = !condition_3 ? matrix_fragment_40 : matrix_fragment_51;

    matrix_fragment_64 = !condition_3 ? matrix_fragment_52 : matrix_fragment_5;

    condition_4 = condition_1 & condition_6;

    matrix_fragment_40 = !condition_3 ? matrix_fragment_51 : matrix_fragment_40;

    matrix_fragment_52 = !condition_3 ? matrix_fragment_5 : matrix_fragment_52;

    matrix_fragment_51 = !condition_2 ? matrix_fragment_68 : matrix_accumulator_2;

    matrix_fragment_6 = !condition_2 ? matrix_fragment_69 : matrix_accumulator_3;

    matrix_fragment_55 = !condition_2 ? matrix_accumulator_2 : matrix_fragment_68;

    matrix_fragment_4 = !condition_2 ? matrix_accumulator_3 : matrix_fragment_69;

    matrix_fragment_56 = !condition_2 ? matrix_fragment_8 : matrix_fragment_70;

    matrix_fragment_5 = !condition_2 ? matrix_fragment_9 : matrix_fragment_71;

    matrix_fragment_39 = !condition_3 ? matrix_accumulator_33 : matrix_fragment_30;

    matrix_fragment_48 = !condition_3 ? matrix_fragment_42 : matrix_fragment_43;

    matrix_accumulator_33 = !condition_3 ? matrix_fragment_30 : matrix_accumulator_33;

    matrix_fragment_42 = !condition_3 ? matrix_fragment_43 : matrix_fragment_42;

    matrix_fragment_43 = !condition_2 ? matrix_fragment_24 : matrix_fragment_14;

    matrix_fragment_30 = !condition_2 ? matrix_fragment_25 : matrix_fragment_15;

    matrix_fragment_62 = !condition_3 ? matrix_fragment_51 : matrix_fragment_6;

    matrix_fragment_59 = !condition_3 ? matrix_fragment_55 : matrix_fragment_4;

    matrix_fragment_67 = !condition_3 ? matrix_fragment_56 : matrix_fragment_5;

    matrix_fragment_51 = !condition_3 ? matrix_fragment_6 : matrix_fragment_51;

    matrix_fragment_55 = !condition_3 ? matrix_fragment_4 : matrix_fragment_55;

    matrix_fragment_56 = !condition_3 ? matrix_fragment_5 : matrix_fragment_56;

    matrix_fragment_57 = !condition_2 ? packed_half_10 : matrix_fragment_76;

    matrix_fragment_53 = !condition_2 ? matrix_fragment_76 : packed_half_10;

    matrix_fragment_6 = !condition_2 ? matrix_fragment_77 : matrix_fragment_11;

    matrix_fragment_4 = !condition_2 ? matrix_fragment_11 : matrix_fragment_77;

    condition_0 = int32_t(matrix_accumulator_32) > int32_t(0u - 0x8u) && !condition_0;

    matrix_fragment_54 = !condition_2 ? matrix_accumulator_16 : matrix_fragment_78;

    matrix_fragment_5 = !condition_2 ? matrix_accumulator_17 : matrix_fragment_79;

    matrix_fragment_58 = !condition_2 ? matrix_fragment_78 : matrix_accumulator_16;

    matrix_fragment_7 = !condition_2 ? matrix_fragment_79 : matrix_accumulator_17;

    matrix_fragment_47 = !condition_3 ? matrix_fragment_43 : matrix_fragment_30;

    matrix_fragment_43 = !condition_3 ? matrix_fragment_30 : matrix_fragment_43;

    matrix_fragment_66 = !condition_3 ? matrix_fragment_53 : matrix_fragment_6;

    matrix_fragment_73 = !condition_3 ? matrix_fragment_57 : matrix_fragment_4;

    condition_2 = condition_0 & condition_6;

    split_address_words(matrix_fragment_30, matrix_fragment_31,
                        int64_t(int32_t(matrix_fragment_31)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_60, matrix_fragment_61));

    matrix_fragment_75 = !condition_3 ? matrix_fragment_54 : matrix_fragment_5;

    matrix_fragment_72 = !condition_3 ? matrix_fragment_58 : matrix_fragment_7;

    matrix_fragment_53 = !condition_3 ? matrix_fragment_6 : matrix_fragment_53;

    matrix_fragment_57 = !condition_3 ? matrix_fragment_4 : matrix_fragment_57;

    condition_1 = condition_1 & condition_5;

    split_address_words(matrix_fragment_60, matrix_fragment_61,
                        int64_t(int32_t(matrix_fragment_63)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_60, matrix_fragment_61));

    condition_0 = condition_0 & condition_5;

    matrix_fragment_54 = !condition_3 ? matrix_fragment_5 : matrix_fragment_54;

    matrix_fragment_58 = !condition_3 ? matrix_fragment_7 : matrix_fragment_58;

    matrix_fragment_65 = 0x4u ^ packed_half_0;
    if (condition_4) {
        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                           select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_24 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_60, matrix_fragment_61));

        matrix_fragment_7 = matrix_fragment_25;

        matrix_fragment_5 = matrix_fragment_24;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_20, matrix_accumulator_21), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    if (condition_1) {
        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_60, matrix_fragment_61,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_60, matrix_fragment_61));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(matrix_fragment_19));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_60, matrix_fragment_61) + 0x1000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    if (condition_2) {
        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                          select_half_operand<2, false, false>(matrix_fragment_9));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_30, matrix_fragment_31));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    if (condition_0) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_30, matrix_fragment_31,
                            int64_t(int32_t(address_word_145)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_30, matrix_fragment_31));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_2 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_3 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                          select_half_operand<2, false, false>(matrix_accumulator_2));

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                          select_half_operand<2, false, false>(matrix_accumulator_3));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_30, matrix_fragment_31) + 0x1000ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    matrix_accumulator_2 = 0x10u ^ packed_half_0;

    __syncthreads();

    matrix_accumulator_3 = 0x14u ^ packed_half_0;

    split_address_words(address_word_158, address_word_159,
                        combine_words(address_word_158, address_word_159) + 0xac640u);

    matrix_fragment_35 = matrix_fragment_35 + 0x1u;

    matrix_fragment_34 = matrix_fragment_34 + 0x1u;

    uniform_block_index_14 = blockIdx.x;

    uniform_index_word_4 = parameters[parameter_optional_control_10];

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_13 = 0u;

    uniform_condition_0 = true;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_64 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_45), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    uniform_index_word_4 = (uniform_block_index_14 << 0x3u) + uniform_index_word_4;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_49), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    uniform_index_word_4 =
        uint32_t(combine_words(uniform_index_word_4, 0u) << 0x1u >> 32) + uniform_index_word_4;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_4) >> 0x1u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_36), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_38), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_52), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_56), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_41), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_50));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_54), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_54 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_58), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_39), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_59), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_59 = result_word;
    }

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(packed_half_0), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_33), "r"(matrix_accumulator_2), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_33 = result_word;
    }

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_37), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_fragment_36 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_44), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_44 = result_word;
    }

    matrix_fragment_36 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_36));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_62), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_62 = result_word;
    }

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_48), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_51), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_55), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_55 = result_word;
    }

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_42), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                        select_half_operand<2, false, false>(matrix_fragment_44));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_43), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    matrix_fragment_59 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_53), "r"(matrix_accumulator_3), "r"(0x1fu), "r"(__activemask()));
        packed_half_0 = result_word;
    }

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_57), "r"(matrix_accumulator_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_fragment_35, 0u) << 0x1u >> 32) + matrix_fragment_35;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_2 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1u);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x34003400u);

    matrix_fragment_5 = matrix_accumulator_2 + 0x3u;

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                           select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 =
        uint32_t(combine_words(matrix_fragment_34, 0u) << 0x1u >> 32) + matrix_fragment_34;

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x34003400u);

    matrix_fragment_6 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);

    matrix_accumulator_33 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_33));

    packed_half_0 = uint32_t(int32_t(packed_half_0) >> 0x1u);

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                   select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_33 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_33), 0x34003400u);

    matrix_fragment_5 =
        uint32_t(combine_words(matrix_fragment_6, 0u) << 0x2u >> 32) + matrix_fragment_5;

    matrix_fragment_4 = packed_half_0 + 0x3u;

    matrix_fragment_6 =
        uint32_t(combine_words(matrix_accumulator_32, 0u) << 0x1u >> 32) + matrix_accumulator_32;

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_33),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_accumulator_3 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    matrix_fragment_6 = uint32_t(int32_t(matrix_fragment_6) >> 0x1u);

    matrix_accumulator_3 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + matrix_fragment_4;

    matrix_fragment_4 = matrix_fragment_5 * 0x4u;

    matrix_fragment_7 = matrix_fragment_5 & 0xfffffffcu;

    matrix_fragment_12 = matrix_fragment_6 + 0x2u;

    matrix_accumulator_3 = matrix_accumulator_3 & 0xfffffffcu;

    __syncthreads();

    store_four_shared_words<CheckBounds>(shared_tile, tile_word_149, matrix_fragment_8,
                                         matrix_fragment_9, packed_half_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    packed_half_10 = thread_row_156 * 0x2u;

    matrix_fragment_11 = matrix_fragment_4 & 0xfffffff0u;

    __syncthreads();
    do {
        uniform_index_word_4 = 0u;

        matrix_fragment_45 = threadIdx.x & 31u;

        matrix_fragment_4 = index_word_148 + matrix_fragment_13;

        uniform_index_word_6 = 0x400u;

        matrix_fragment_14 = 0;
        matrix_fragment_15 = 0;

        matrix_accumulator_16 = 0;
        matrix_accumulator_17 = 0;

        matrix_fragment_18 = 0;
        matrix_fragment_19 = 0;

        matrix_fragment_5 = matrix_fragment_4 * 0x8u;

        matrix_accumulator_20 = 0;
        matrix_accumulator_21 = 0;

        uniform_condition_1 = uniform_condition_0;

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            uint64_t(uint32_t(matrix_fragment_5)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        uniform_index_word_6 = (uniform_index_word_4 << 0x18u) + uniform_index_word_6;

        uniform_index_word_4 = 0u;
        do {
            split_address_words(matrix_fragment_8, matrix_fragment_9,
                                int64_t(int32_t(matrix_fragment_45)) * int64_t(int32_t(0x10u)) +
                                    combine_words(matrix_fragment_4, matrix_fragment_5));

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_8, matrix_fragment_9) - 0x4200ull, matrix_fragment_24,
                matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
                error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_8, matrix_fragment_9) - 0x4000ull, matrix_fragment_28,
                matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
                error_counts);

            load_four_words_cached<CheckBounds>(
                combine_words(matrix_fragment_8, matrix_fragment_9) - 0x200ull, matrix_fragment_36,
                matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
                error_counts);

            load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                                matrix_fragment_40, matrix_fragment_41,
                                                matrix_fragment_42, matrix_fragment_43,
                                                memory_ranges, error_counts);

            matrix_accumulator_22 = (matrix_fragment_45 << 0x4u) + uniform_index_word_6;

            uniform_condition_0 = uint32_t(uniform_index_word_4) >= uint32_t(0xc0u);

            uniform_parameter_word_7 = uniform_index_word_4 + 0x40u;

            split_address_words(matrix_fragment_4, matrix_fragment_5,
                                combine_words(matrix_fragment_4, matrix_fragment_5) + 0x8000u);

            load_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_22, matrix_accumulator_32, matrix_accumulator_33,
                matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

            uniform_index_word_6 = uniform_index_word_6 + 0x400u;

            uniform_index_word_4 = uniform_parameter_word_7;

            multiply_accumulate_half(
                matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_32,
                matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35, matrix_fragment_24,
                matrix_fragment_25, matrix_accumulator_20, matrix_accumulator_21);

            multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_accumulator_32,
                                     matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_26, matrix_fragment_27, matrix_fragment_18,
                                     matrix_fragment_19);

            load_four_shared_words<CheckBounds>(
                shared_tile, matrix_accumulator_22 + 0x200u, matrix_fragment_24, matrix_fragment_25,
                matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

            multiply_accumulate_half(
                matrix_accumulator_16, matrix_accumulator_17, matrix_accumulator_32,
                matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35, matrix_fragment_28,
                matrix_fragment_29, matrix_accumulator_16, matrix_accumulator_17);

            multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_accumulator_32,
                                     matrix_accumulator_33, matrix_fragment_34, matrix_fragment_35,
                                     matrix_fragment_30, matrix_fragment_31, matrix_fragment_14,
                                     matrix_fragment_15);

            multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21,
                                     matrix_fragment_24, matrix_fragment_25, matrix_fragment_26,
                                     matrix_fragment_27, matrix_fragment_36, matrix_fragment_37,
                                     matrix_accumulator_20, matrix_accumulator_21);

            multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_24,
                                     matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                     matrix_fragment_38, matrix_fragment_39, matrix_fragment_18,
                                     matrix_fragment_19);

            multiply_accumulate_half(matrix_accumulator_16, matrix_accumulator_17,
                                     matrix_fragment_24, matrix_fragment_25, matrix_fragment_26,
                                     matrix_fragment_27, matrix_fragment_40, matrix_fragment_41,
                                     matrix_accumulator_16, matrix_accumulator_17);

            multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_24,
                                     matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                     matrix_fragment_42, matrix_fragment_43, matrix_fragment_14,
                                     matrix_fragment_15);
        } while (!uniform_condition_0);
        matrix_fragment_25 = matrix_fragment_45;

        matrix_fragment_5 = matrix_fragment_45;

        matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_25) >> 0x2u);

        matrix_fragment_8 = uint32_t(int64_t(int32_t(matrix_fragment_5)) * (1ll << 0x1cu) >> 32) +
                            matrix_fragment_6;

        matrix_fragment_4 =
            uint32_t(combine_words(matrix_fragment_9, 0u) << 0x2u >> 32) + matrix_fragment_9;

        matrix_fragment_26 = uint32_t(int64_t(int32_t(matrix_fragment_5)) * (1ll << 0x1cu) >> 32) +
                             matrix_fragment_12;

        matrix_fragment_4 = matrix_fragment_4 & 0xfffffffcu;

        condition_0 = int32_t(matrix_fragment_8) >= int32_t(matrix_accumulator_3);

        condition_2 = int32_t(matrix_fragment_26) >= int32_t(matrix_accumulator_3);

        matrix_accumulator_22 = matrix_fragment_9 + uniform_index_word_5 + (0u - matrix_fragment_4);

        matrix_accumulator_21 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                          select_half_operand<2, false, false>(0u));

        condition_0 = int32_t(matrix_fragment_8) < int32_t(0u) || condition_0;

        condition_1 = int32_t(matrix_accumulator_22) >= int32_t(matrix_fragment_7);

        condition_2 = int32_t(matrix_fragment_26) < int32_t(0u) || condition_2;

        matrix_accumulator_23 =
            uint32_t(combine_words(matrix_fragment_13, 0u) << 0x1cu >> 32) + packed_half_10;

        condition_1 = int32_t(matrix_accumulator_22) < int32_t(0u) || condition_1;

        condition_0 = condition_0 | condition_1;

        condition_2 = condition_2 | condition_1;
        if (!condition_0) {
            matrix_fragment_4 = parameters[parameter_output_downsample_optional_low_18];
            matrix_fragment_5 = parameters[parameter_output_downsample_optional_high_19];

            matrix_fragment_24 = matrix_accumulator_22 * 0x4u;

            matrix_fragment_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_9 = matrix_accumulator_3 * matrix_accumulator_23 + matrix_fragment_8;

            matrix_fragment_13 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                              select_half_operand<2, false, false>(matrix_fragment_13));

            matrix_fragment_9 = matrix_fragment_11 * matrix_fragment_9 + matrix_fragment_24;

            matrix_fragment_24 = matrix_fragment_25 & 0xfffffffcu;

            matrix_fragment_9 = matrix_fragment_9 + matrix_fragment_45 + (0u - matrix_fragment_24);

            split_address_words(matrix_fragment_4, matrix_fragment_5,
                                int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_4, matrix_fragment_5));

            pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                        matrix_fragment_13, memory_ranges, error_counts);
        }
        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_16),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                          select_half_operand<2, false, false>(0u));
        if (!condition_2) {
            matrix_fragment_4 = parameters[parameter_output_downsample_optional_low_18];
            matrix_fragment_5 = parameters[parameter_output_downsample_optional_high_19];

            matrix_fragment_9 = matrix_accumulator_22 * 0x4u;

            matrix_fragment_14 = matrix_accumulator_3 * matrix_accumulator_23 + matrix_fragment_26;

            matrix_fragment_14 = matrix_fragment_11 * matrix_fragment_14 + matrix_fragment_9;

            matrix_fragment_9 = matrix_fragment_25 & 0xfffffffcu;

            matrix_fragment_9 = matrix_fragment_14 + matrix_fragment_45 + (0u - matrix_fragment_9);

            split_address_words(matrix_fragment_4, matrix_fragment_5,
                                int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_4, matrix_fragment_5));

            matrix_fragment_9 =
                permute_signed(matrix_accumulator_21, matrix_accumulator_16, 0x5410u);

            pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        matrix_accumulator_23 = matrix_accumulator_23 + 0x1u;
        if (!condition_0) {
            matrix_fragment_4 = parameters[parameter_output_downsample_optional_low_18];
            matrix_fragment_5 = parameters[parameter_output_downsample_optional_high_19];

            matrix_fragment_9 = matrix_accumulator_22 * 0x4u;

            matrix_fragment_8 = matrix_accumulator_3 * matrix_accumulator_23 + matrix_fragment_8;

            matrix_fragment_8 = matrix_fragment_11 * matrix_fragment_8 + matrix_fragment_9;

            matrix_fragment_9 = matrix_fragment_25 & 0xfffffffcu;

            matrix_fragment_9 = matrix_fragment_8 + matrix_fragment_45 + (0u - matrix_fragment_9);

            split_address_words(matrix_fragment_4, matrix_fragment_5,
                                int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_4, matrix_fragment_5));

            matrix_fragment_9 = permute_signed(matrix_fragment_13, matrix_fragment_18, 0x5410u);

            pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        matrix_accumulator_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(0u));
        if (!condition_2) {
            matrix_fragment_4 = parameters[parameter_output_downsample_optional_low_18];
            matrix_fragment_5 = parameters[parameter_output_downsample_optional_high_19];

            matrix_accumulator_22 = matrix_accumulator_22 * 0x4u;

            matrix_fragment_9 = matrix_fragment_25 & 0xfffffffcu;

            matrix_fragment_26 = matrix_accumulator_3 * matrix_accumulator_23 + matrix_fragment_26;

            matrix_fragment_26 = matrix_fragment_11 * matrix_fragment_26 + matrix_accumulator_22;

            matrix_fragment_9 = matrix_fragment_26 + matrix_fragment_45 + (0u - matrix_fragment_9);

            split_address_words(matrix_fragment_4, matrix_fragment_5,
                                int64_t(int32_t(matrix_fragment_9)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_4, matrix_fragment_5));

            matrix_fragment_9 = permute_signed(matrix_accumulator_17, matrix_fragment_8, 0x5410u);

            pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        matrix_fragment_13 = 0x100u;

        uniform_condition_0 = false;
    } while (uniform_condition_1);
    uniform_index_word_6 = parameters[parameter_output_downsample_optional_low_18];
    uniform_parameter_word_7 = parameters[parameter_output_downsample_optional_high_19];

    condition_0 = 0ull != combine_words(uniform_index_word_6, uniform_parameter_word_7);

    if (!condition_0) {
        return;
    }

    uniform_index_word_6 = parameters[parameter_tensor_height_20];
    uniform_parameter_word_7 = parameters[parameter_tensor_width_21];

    matrix_fragment_9 = uniform_parameter_word_7;

    matrix_fragment_19 = uniform_index_word_6;

    condition_0 = int32_t(matrix_accumulator_2) >= int32_t(matrix_fragment_9);

    condition_1 = int32_t(packed_half_0) >= int32_t(matrix_fragment_19);

    if (condition_0) {
        return;
    }

    uniform_parameter_word_12 = blockDim.x;

    matrix_fragment_13 = threadIdx.z;

    matrix_accumulator_3 = matrix_fragment_6 + 0x4u;

    matrix_fragment_4 = matrix_fragment_9 * 0x4u;

    uniform_reused_word_13 = blockDim.y;

    matrix_accumulator_17 = blockDim.z;

    matrix_fragment_12 = threadIdx.x;

    condition_0 = int32_t(matrix_accumulator_3) > int32_t(packed_half_0);

    uniform_parameter_word_8 = parameters[parameter_optional_control_10];
    uniform_parameter_word_9 = parameters[parameter_scalar_control_11];

    matrix_accumulator_3 = matrix_fragment_4 * matrix_fragment_19;

    uniform_index_word_4 = gridDim.y;

    condition_0 = int32_t(matrix_fragment_6) < int32_t(matrix_fragment_19) && condition_0;

    uniform_index_word_6 = gridDim.x;

    uniform_parameter_word_7 = uniform_index_word_5 + 0x4u;

    condition_1 = int32_t(matrix_accumulator_2) < int32_t(uniform_parameter_word_7);

    uniform_parameter_word_7 = uniform_parameter_word_12 * uniform_reused_word_13;

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_parameter_word_9;

    condition_1 = int32_t(matrix_fragment_9) > int32_t(uniform_index_word_5) && condition_1;

    matrix_fragment_8 = matrix_fragment_13 * uniform_reused_word_13 + thread_row_156;

    uniform_index_word_4 = uniform_index_word_4 + (0u - 0x8u);

    uniform_index_word_6 = (uniform_index_word_6 << 0x3u) + uniform_parameter_word_8;

    condition_0 = condition_0 | condition_1;

    matrix_fragment_5 = matrix_accumulator_17 * uniform_parameter_word_7;

    uniform_index_word_4 =
        uint32_t(combine_words(uniform_index_word_4, 0u) << 0x1u >> 32) + uniform_index_word_4;

    uniform_index_word_6 = uniform_index_word_6 + (0u - 0x8u);

    matrix_fragment_8 = matrix_fragment_8 * uniform_parameter_word_12 + matrix_fragment_12;

    uniform_index_word_4 =
        uint32_t(int64_t(int32_t(uniform_index_word_4)) * (1ll << 0x1fu) >> 32) + 0x4u;

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    uniform_index_word_6 =
        uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1fu) >> 32) + 0x4u;

    matrix_fragment_7 = min(int32_t(matrix_fragment_19), int32_t(uniform_index_word_4));
    if (condition_0) {
        __syncthreads();

        condition_0 = uint32_t(matrix_fragment_8) > uint32_t(0x3ffu);

        uniform_parameter_word_8 = parameters[parameter_output_downsample_optional_low_18];
        uniform_parameter_word_9 = parameters[parameter_output_downsample_optional_high_19];

        uniform_index_word_16 = parameters[parameter_output_downsample_optional_low_18];
        uniform_address_word_17 = parameters[parameter_output_downsample_optional_high_19];
        if (!condition_0) {
            const uint32_t next_boundary_index =
                ((matrix_fragment_13 + matrix_accumulator_17) * uniform_reused_word_13 + thread_row_156) *
                    uniform_parameter_word_12 + matrix_fragment_12;
            matrix_accumulator_16 = matrix_fragment_8;
            matrix_fragment_14 = boundary_tail_steps<false>(next_boundary_index, 1024u, matrix_fragment_5);

            packed_half_10 = matrix_fragment_14 + 0x1u;

            packed_half_10 = packed_half_10 & 0x3u;
            condition_0 = packed_half_10 != 0;
            if (condition_0) {
                matrix_accumulator_20 = parameters[parameter_tensor_height_20];
                matrix_accumulator_21 = parameters[parameter_tensor_width_21];

                matrix_fragment_15 = 0u - packed_half_10;

                matrix_accumulator_16 = matrix_fragment_8;
                do {
                    matrix_fragment_9 = uint32_t(uint32_t(matrix_accumulator_16) >> 0x6u);

                    matrix_fragment_13 =
                        uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x18u >> 32) +
                        matrix_fragment_6;

                    matrix_fragment_9 = matrix_fragment_9 & 0x3u;

                    matrix_fragment_19 = matrix_accumulator_20;

                    matrix_fragment_15 = matrix_fragment_15 + 0x1u;

                    matrix_fragment_12 = matrix_fragment_9 + uniform_index_word_5;

                    matrix_fragment_9 = matrix_accumulator_21;

                    condition_0 = int32_t(matrix_fragment_13) >= int32_t(matrix_fragment_19);

                    condition_2 = int32_t(matrix_fragment_12) >= int32_t(matrix_accumulator_2);

                    condition_1 = int32_t(matrix_fragment_12) >= int32_t(matrix_fragment_9);

                    condition_0 = int32_t(matrix_fragment_13) < int32_t(0u) || condition_0;

                    condition_2 =
                        int32_t(matrix_fragment_13) < int32_t(packed_half_0) && !condition_2;

                    condition_1 = int32_t(matrix_fragment_12) < int32_t(0u) || condition_1;

                    condition_0 = condition_2 | condition_0 | condition_1;

                    condition_1 = uint32_t(matrix_fragment_15) != 0u;
                    if (!condition_0) {
                        packed_half_10 = matrix_accumulator_16 & 0x3fu;

                        matrix_fragment_12 = matrix_fragment_12 * 0x4u;

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(packed_half_10)) *
                                                    int64_t(int32_t(matrix_accumulator_3)) +
                                                0ull);

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(matrix_fragment_4)) *
                                                    int64_t(int32_t(matrix_fragment_13)) +
                                                combine_words(packed_half_10, matrix_fragment_11));

                        matrix_fragment_13 = 0u;

                        split_address_words(
                            packed_half_10, matrix_fragment_11,
                            combine_words(packed_half_10, matrix_fragment_11) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

                        const uint64_t byte_address_0 = scaled_word_address(
                            combine_words(uniform_parameter_word_8, uniform_parameter_word_9),
                            combine_words(packed_half_10, matrix_fragment_11));
                        matrix_fragment_12 = uint32_t(byte_address_0);

                        matrix_fragment_13 = uint32_t(byte_address_0 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    matrix_accumulator_16 = matrix_fragment_5 + matrix_accumulator_16;
                } while (condition_1);
            }
            condition_0 = uint32_t(matrix_fragment_14) >= uint32_t(0x3u);
            if (condition_0) {
                matrix_fragment_28 = parameters[parameter_tensor_height_20];
                matrix_fragment_29 = parameters[parameter_tensor_width_21];

                matrix_fragment_9 = matrix_accumulator_17 * uniform_reused_word_13;

                matrix_fragment_18 = matrix_fragment_5 + matrix_accumulator_16;

                matrix_fragment_15 = matrix_fragment_9 * uniform_parameter_word_12;

                matrix_fragment_14 = matrix_fragment_15 * 0x2u + matrix_accumulator_16;

                matrix_accumulator_17 = matrix_fragment_15 * 0x3u + matrix_accumulator_16;
                do {
                    packed_half_10 = uint32_t(uint32_t(matrix_accumulator_16) >> 0x6u);

                    matrix_accumulator_21 =
                        uint32_t(combine_words(matrix_accumulator_16, 0u) << 0x18u >> 32) +
                        matrix_fragment_6;

                    packed_half_10 = packed_half_10 & 0x3u;

                    matrix_fragment_19 = matrix_fragment_28;

                    matrix_fragment_9 = matrix_fragment_29;

                    matrix_fragment_11 = uint32_t(uint32_t(matrix_fragment_18) >> 0x6u);

                    matrix_accumulator_20 = packed_half_10 + uniform_index_word_5;

                    matrix_fragment_12 = uint32_t(uint32_t(matrix_fragment_14) >> 0x6u);

                    condition_2 = int32_t(matrix_accumulator_21) >= int32_t(matrix_fragment_19);

                    matrix_fragment_11 = matrix_fragment_11 & 0x3u;

                    condition_3 = int32_t(matrix_accumulator_20) >= int32_t(matrix_accumulator_2);

                    condition_4 = int32_t(matrix_accumulator_20) >= int32_t(matrix_fragment_9);

                    matrix_fragment_12 = matrix_fragment_12 & 0x3u;

                    condition_2 = int32_t(matrix_accumulator_21) < int32_t(0u) || condition_2;

                    condition_3 =
                        int32_t(matrix_accumulator_21) < int32_t(packed_half_0) && !condition_3;

                    condition_4 = int32_t(matrix_accumulator_20) < int32_t(0u) || condition_4;

                    matrix_accumulator_23 =
                        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x18u >> 32) +
                        matrix_fragment_6;

                    matrix_fragment_25 =
                        uint32_t(combine_words(matrix_fragment_14, 0u) << 0x18u >> 32) +
                        matrix_fragment_6;

                    matrix_accumulator_22 = matrix_fragment_11 + uniform_index_word_5;

                    matrix_fragment_24 = matrix_fragment_12 + uniform_index_word_5;

                    matrix_fragment_13 = uint32_t(uint32_t(matrix_accumulator_17) >> 0x6u);

                    condition_2 = condition_3 | condition_2 | condition_4;

                    condition_0 = int32_t(matrix_accumulator_23) >= int32_t(matrix_fragment_19);

                    condition_1 = int32_t(matrix_fragment_25) >= int32_t(matrix_fragment_19);

                    condition_5 = int32_t(matrix_accumulator_22) >= int32_t(matrix_fragment_9);

                    condition_6 = int32_t(matrix_accumulator_22) >= int32_t(matrix_accumulator_2);

                    condition_3 = int32_t(matrix_fragment_24) >= int32_t(matrix_fragment_9);

                    condition_4 = int32_t(matrix_fragment_24) >= int32_t(matrix_accumulator_2);

                    matrix_fragment_13 = matrix_fragment_13 & 0x3u;

                    condition_0 = int32_t(matrix_accumulator_23) < int32_t(0u) || condition_0;

                    condition_1 = int32_t(matrix_fragment_25) < int32_t(0u) || condition_1;

                    condition_5 = int32_t(matrix_accumulator_22) < int32_t(0u) || condition_5;

                    condition_6 =
                        int32_t(matrix_accumulator_23) < int32_t(packed_half_0) && !condition_6;

                    condition_3 = int32_t(matrix_fragment_24) < int32_t(0u) || condition_3;

                    condition_4 =
                        int32_t(matrix_fragment_25) < int32_t(packed_half_0) && !condition_4;

                    matrix_fragment_27 =
                        uint32_t(combine_words(matrix_accumulator_17, 0u) << 0x18u >> 32) +
                        matrix_fragment_6;

                    matrix_fragment_26 = matrix_fragment_13 + uniform_index_word_5;

                    condition_0 = condition_6 | condition_0 | condition_5;

                    condition_1 = condition_4 | condition_1 | condition_3;

                    condition_5 = int32_t(matrix_fragment_27) >= int32_t(matrix_fragment_19);

                    condition_6 = int32_t(matrix_fragment_26) >= int32_t(matrix_fragment_9);

                    condition_3 = int32_t(matrix_fragment_26) >= int32_t(matrix_accumulator_2);

                    condition_5 = int32_t(matrix_fragment_27) < int32_t(0u) || condition_5;

                    condition_6 = int32_t(matrix_fragment_26) < int32_t(0u) || condition_6;

                    condition_3 =
                        int32_t(matrix_fragment_27) < int32_t(packed_half_0) && !condition_3;

                    condition_3 = condition_3 | condition_5 | condition_6;
                    if (!condition_2) {
                        packed_half_10 = matrix_accumulator_16 & 0x3fu;

                        matrix_fragment_12 = matrix_accumulator_20 * 0x4u;

                        matrix_fragment_13 = 0u;

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(packed_half_10)) *
                                                    int64_t(int32_t(matrix_accumulator_3)) +
                                                0ull);

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(matrix_fragment_4)) *
                                                    int64_t(int32_t(matrix_accumulator_21)) +
                                                combine_words(packed_half_10, matrix_fragment_11));

                        split_address_words(
                            packed_half_10, matrix_fragment_11,
                            combine_words(packed_half_10, matrix_fragment_11) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

                        const uint64_t byte_address_1 = scaled_word_address(
                            combine_words(uniform_index_word_16, uniform_address_word_17),
                            combine_words(packed_half_10, matrix_fragment_11));
                        matrix_fragment_12 = uint32_t(byte_address_1);

                        matrix_fragment_13 = uint32_t(byte_address_1 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    if (!condition_0) {
                        packed_half_10 = matrix_fragment_18 & 0x3fu;

                        matrix_fragment_12 = matrix_accumulator_22 * 0x4u;

                        matrix_fragment_13 = 0u;

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(packed_half_10)) *
                                                    int64_t(int32_t(matrix_accumulator_3)) +
                                                0ull);

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(matrix_fragment_4)) *
                                                    int64_t(int32_t(matrix_accumulator_23)) +
                                                combine_words(packed_half_10, matrix_fragment_11));

                        split_address_words(
                            packed_half_10, matrix_fragment_11,
                            combine_words(packed_half_10, matrix_fragment_11) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

                        const uint64_t byte_address_2 = scaled_word_address(
                            combine_words(uniform_index_word_16, uniform_address_word_17),
                            combine_words(packed_half_10, matrix_fragment_11));
                        matrix_fragment_12 = uint32_t(byte_address_2);

                        matrix_fragment_13 = uint32_t(byte_address_2 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    if (!condition_1) {
                        packed_half_10 = matrix_fragment_14 & 0x3fu;

                        matrix_fragment_12 = matrix_fragment_24 * 0x4u;

                        matrix_fragment_13 = 0u;

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(packed_half_10)) *
                                                    int64_t(int32_t(matrix_accumulator_3)) +
                                                0ull);

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(matrix_fragment_4)) *
                                                    int64_t(int32_t(matrix_fragment_25)) +
                                                combine_words(packed_half_10, matrix_fragment_11));

                        split_address_words(
                            packed_half_10, matrix_fragment_11,
                            combine_words(packed_half_10, matrix_fragment_11) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

                        const uint64_t byte_address_3 = scaled_word_address(
                            combine_words(uniform_index_word_16, uniform_address_word_17),
                            combine_words(packed_half_10, matrix_fragment_11));
                        matrix_fragment_12 = uint32_t(byte_address_3);

                        matrix_fragment_13 = uint32_t(byte_address_3 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    matrix_accumulator_16 =
                        matrix_fragment_5 + matrix_accumulator_16 + matrix_fragment_5;
                    if (!condition_3) {
                        packed_half_10 = matrix_accumulator_17 & 0x3fu;

                        matrix_fragment_12 = matrix_fragment_26 * 0x4u;

                        matrix_fragment_13 = 0u;

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(packed_half_10)) *
                                                    int64_t(int32_t(matrix_accumulator_3)) +
                                                0ull);

                        split_address_words(packed_half_10, matrix_fragment_11,
                                            int64_t(int32_t(matrix_fragment_4)) *
                                                    int64_t(int32_t(matrix_fragment_27)) +
                                                combine_words(packed_half_10, matrix_fragment_11));

                        split_address_words(
                            packed_half_10, matrix_fragment_11,
                            combine_words(packed_half_10, matrix_fragment_11) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));

                        const uint64_t byte_address_4 = scaled_word_address(
                            combine_words(uniform_index_word_16, uniform_address_word_17),
                            combine_words(packed_half_10, matrix_fragment_11));
                        matrix_fragment_12 = uint32_t(byte_address_4);

                        matrix_fragment_13 = uint32_t(byte_address_4 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    matrix_accumulator_16 =
                        matrix_fragment_5 + matrix_accumulator_16 + matrix_fragment_5;

                    matrix_fragment_14 = matrix_fragment_15 * 0x4u + matrix_fragment_14;

                    matrix_accumulator_17 = matrix_fragment_15 * 0x4u + matrix_accumulator_17;

                    condition_0 = uint32_t(matrix_accumulator_16) >= uint32_t(0x400u);

                    matrix_fragment_18 = matrix_fragment_15 * 0x4u + matrix_fragment_18;
                } while (!condition_0);
            }
        }
    }
    matrix_fragment_6 = blockIdx.z;

    matrix_fragment_11 = blockIdx.y;

    condition_0 = (matrix_fragment_11 | matrix_fragment_6 | uniform_block_index_14) != 0;

    if (condition_0) {
        return;
    }

    packed_half_0 = max(int32_t(packed_half_0), int32_t(matrix_fragment_7));

    matrix_fragment_11 = min(int32_t(matrix_fragment_9), int32_t(uniform_index_word_6));

    uniform_parameter_word_7 = parameters[parameter_tensor_width_21];

    matrix_fragment_6 = 0u - packed_half_0 + matrix_fragment_19;

    uniform_parameter_word_12 = parameters[parameter_tensor_width_21];

    matrix_accumulator_2 = max(int32_t(matrix_accumulator_2), int32_t(matrix_fragment_11));

    condition_0 = int32_t(matrix_fragment_6) >= int32_t(0x1u);

    uniform_index_word_4 = parameters[parameter_output_downsample_optional_low_18];
    uniform_index_word_5 = parameters[parameter_output_downsample_optional_high_19];

    uniform_parameter_word_8 = parameters[parameter_output_downsample_optional_low_18];
    uniform_parameter_word_9 = parameters[parameter_output_downsample_optional_high_19];
    if (condition_0) {
        matrix_fragment_11 = matrix_fragment_6 * matrix_fragment_9;

        matrix_fragment_11 = matrix_fragment_11 * 0x40u;

        condition_0 = int32_t(matrix_fragment_8) >= int32_t(matrix_fragment_11);
        if (!condition_0) {
            packed_half_10 = boundary_tail_steps<true>(
                matrix_fragment_5 + matrix_fragment_8, matrix_fragment_11, matrix_fragment_5);

            matrix_fragment_6 = packed_half_10 + 0x1u;

            condition_0 = uint32_t(packed_half_10) >= uint32_t(0x3u);

            matrix_fragment_14 = matrix_fragment_6 & 0x3u;
            condition_1 = matrix_fragment_14 != 0;

            matrix_fragment_6 = matrix_fragment_8;
            if (condition_1) {
                matrix_fragment_18 =
                    signed_word_magnitude(matrix_fragment_9);

                condition_2 = int32_t(matrix_fragment_9) != int32_t(0u);

                packed_half_10 = 0u - matrix_fragment_14;

                matrix_fragment_6 = __float_as_uint(__int2float_ru(matrix_fragment_18));

                matrix_accumulator_17 = ~matrix_fragment_9;

                matrix_fragment_6 = reciprocal_bits(matrix_fragment_6);

                matrix_fragment_12 = matrix_fragment_6 + 0xffffffeu;

                matrix_fragment_6 = matrix_fragment_8;

                matrix_fragment_13 = __float2uint_rz(__uint_as_float(matrix_fragment_12));

                matrix_fragment_12 = 0u;

                matrix_fragment_15 = 0u - matrix_fragment_13;

                matrix_fragment_15 = matrix_fragment_15 * matrix_fragment_18;

                matrix_accumulator_16 =
                    uint32_t(uint64_t(matrix_fragment_13) * matrix_fragment_15 +
                                 combine_words(matrix_fragment_12, matrix_fragment_13) >>
                             32);
                do {
                    matrix_accumulator_20 = uint32_t(uint32_t(matrix_fragment_6) >> 0x6u);

                    matrix_fragment_9 = uniform_parameter_word_7;

                    packed_half_10 = packed_half_10 + 0x1u;

                    matrix_fragment_13 = int32_t(matrix_accumulator_20) < 0
                                             ? 0u - matrix_accumulator_20
                                             : matrix_accumulator_20;

                    matrix_accumulator_22 =
                        signed_word_magnitude(matrix_fragment_9);

                    matrix_fragment_12 =
                        uint32_t(uint64_t(matrix_accumulator_16) * matrix_fragment_13 >> 32);

                    matrix_fragment_14 = 0u - matrix_fragment_12;

                    matrix_fragment_13 =
                        matrix_accumulator_22 * matrix_fragment_14 + matrix_fragment_13;

                    matrix_fragment_14 = matrix_fragment_6 & 0x3fu;

                    matrix_fragment_6 = matrix_fragment_5 + matrix_fragment_6;

                    condition_3 = uint32_t(matrix_fragment_18) > uint32_t(matrix_fragment_13);

                    if (!condition_3) {
                        matrix_fragment_13 = matrix_fragment_13 + (0u - matrix_accumulator_22);

                        matrix_fragment_12 = matrix_fragment_12 + 0x1u;
                    }

                    condition_1 = uint32_t(matrix_fragment_13) >= uint32_t(matrix_fragment_18);

                    matrix_fragment_13 = matrix_accumulator_20 ^ matrix_fragment_9;

                    condition_3 = int32_t(matrix_fragment_13) >= int32_t(0u);

                    if (condition_1) {
                        matrix_fragment_12 = matrix_fragment_12 + 0x1u;
                    }

                    if (!condition_3) {
                        matrix_fragment_12 = 0u - matrix_fragment_12;
                    }

                    matrix_fragment_15 = !condition_2 ? matrix_accumulator_17 : matrix_fragment_12;

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        int64_t(int32_t(matrix_fragment_14)) *
                                                int64_t(int32_t(matrix_accumulator_3)) +
                                            0ull);

                    matrix_fragment_19 = packed_half_0 + matrix_fragment_15;

                    matrix_fragment_14 = 0u - matrix_fragment_15;

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        int64_t(int32_t(matrix_fragment_4)) *
                                                int64_t(int32_t(matrix_fragment_19)) +
                                            combine_words(matrix_fragment_12, matrix_fragment_13));

                    matrix_fragment_15 =
                        matrix_fragment_9 * matrix_fragment_14 + matrix_accumulator_20;

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        uint64_t(uint32_t(matrix_fragment_15)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_12, matrix_fragment_13));

                    const uint64_t byte_address_5 = scaled_word_address(
                        combine_words(uniform_index_word_4, uniform_index_word_5),
                        combine_words(matrix_fragment_12, matrix_fragment_13));
                    matrix_fragment_14 = uint32_t(byte_address_5);

                    matrix_fragment_15 = uint32_t(byte_address_5 >> 32);

                    condition_1 = uint32_t(packed_half_10) != 0u;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15), 0u, memory_ranges,
                        error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xcull, 0u,
                        memory_ranges, error_counts);
                } while (condition_1);
            }
            if (condition_0) {
                matrix_fragment_31 =
                    signed_word_magnitude(matrix_fragment_9);

                matrix_fragment_9 = __float_as_uint(__int2float_ru(matrix_fragment_31));

                matrix_fragment_9 = reciprocal_bits(matrix_fragment_9);

                matrix_fragment_12 = matrix_fragment_9 + 0xffffffeu;

                matrix_fragment_13 = __float2uint_rz(__uint_as_float(matrix_fragment_12));

                matrix_fragment_12 = 0u;

                packed_half_10 = 0u - matrix_fragment_13;

                matrix_fragment_15 = packed_half_10 * matrix_fragment_31;

                matrix_accumulator_33 =
                    uint32_t(uint64_t(matrix_fragment_13) * matrix_fragment_15 +
                                 combine_words(matrix_fragment_12, matrix_fragment_13) >>
                             32);
                do {
                    matrix_fragment_9 = uniform_parameter_word_12;

                    matrix_fragment_24 = uint32_t(uint32_t(matrix_fragment_6) >> 0x6u);

                    matrix_fragment_26 = matrix_fragment_5 + matrix_fragment_6;

                    matrix_accumulator_23 =
                        signed_word_magnitude(matrix_fragment_9);

                    matrix_fragment_14 = int32_t(matrix_fragment_24) < 0 ? 0u - matrix_fragment_24
                                                                         : matrix_fragment_24;

                    matrix_accumulator_16 = __float_as_uint(__int2float_ru(matrix_accumulator_23));

                    matrix_fragment_28 = uint32_t(uint32_t(matrix_fragment_26) >> 0x6u);

                    packed_half_10 =
                        uint32_t(uint64_t(matrix_accumulator_33) * matrix_fragment_14 >> 32);

                    matrix_fragment_30 = matrix_fragment_5 + matrix_fragment_26;

                    matrix_fragment_27 = ~matrix_fragment_9;

                    matrix_fragment_15 = 0u - packed_half_10;

                    matrix_fragment_34 = matrix_fragment_5 + matrix_fragment_30;

                    matrix_accumulator_32 = uint32_t(uint32_t(matrix_fragment_30) >> 0x6u);

                    matrix_fragment_14 =
                        matrix_accumulator_23 * matrix_fragment_15 + matrix_fragment_14;

                    matrix_fragment_36 = uint32_t(uint32_t(matrix_fragment_34) >> 0x6u);

                    matrix_accumulator_16 = reciprocal_bits(matrix_accumulator_16);

                    matrix_fragment_19 = int32_t(matrix_accumulator_32) < 0
                                             ? 0u - matrix_accumulator_32
                                             : matrix_accumulator_32;

                    condition_1 = uint32_t(matrix_fragment_31) > uint32_t(matrix_fragment_14);

                    matrix_accumulator_21 = int32_t(matrix_fragment_36) < 0
                                                ? 0u - matrix_fragment_36
                                                : matrix_fragment_36;

                    if (!condition_1) {
                        matrix_fragment_14 = matrix_fragment_14 + (0u - matrix_accumulator_23);

                        packed_half_10 = packed_half_10 + 0x1u;
                    }

                    matrix_fragment_12 = matrix_accumulator_16 + 0xffffffeu;

                    condition_0 = uint32_t(matrix_fragment_14) >= uint32_t(matrix_fragment_31);

                    matrix_fragment_13 = __float2uint_rz(__uint_as_float(matrix_fragment_12));

                    matrix_fragment_31 = matrix_accumulator_23;

                    matrix_fragment_12 = 0u;

                    if (condition_0) {
                        packed_half_10 = packed_half_10 + 0x1u;
                    }

                    condition_0 = int32_t(matrix_fragment_9) != int32_t(0u);

                    matrix_accumulator_16 = 0u - matrix_fragment_13;

                    matrix_accumulator_33 = matrix_accumulator_16 * matrix_accumulator_23;

                    matrix_accumulator_16 = int32_t(matrix_fragment_28) < 0
                                                ? 0u - matrix_fragment_28
                                                : matrix_fragment_28;

                    matrix_accumulator_33 =
                        uint32_t(uint64_t(matrix_fragment_13) * matrix_accumulator_33 +
                                     combine_words(matrix_fragment_12, matrix_fragment_13) >>
                                 32);

                    matrix_fragment_12 = matrix_fragment_24 ^ matrix_fragment_9;

                    matrix_fragment_14 =
                        uint32_t(uint64_t(matrix_accumulator_33) * matrix_accumulator_16 >> 32);

                    condition_3 = int32_t(matrix_fragment_12) >= int32_t(0u);

                    matrix_fragment_18 =
                        uint32_t(uint64_t(matrix_accumulator_33) * matrix_fragment_19 >> 32);

                    matrix_fragment_13 = 0u - matrix_fragment_14;

                    matrix_accumulator_20 =
                        uint32_t(uint64_t(matrix_accumulator_33) * matrix_accumulator_21 >> 32);

                    matrix_fragment_12 = 0u - matrix_fragment_18;

                    matrix_accumulator_16 =
                        matrix_accumulator_23 * matrix_fragment_13 + matrix_accumulator_16;

                    matrix_accumulator_22 = 0u - matrix_accumulator_20;

                    matrix_fragment_19 =
                        matrix_accumulator_23 * matrix_fragment_12 + matrix_fragment_19;

                    if (!condition_3) {
                        packed_half_10 = 0u - packed_half_10;
                    }

                    matrix_fragment_12 = matrix_fragment_6 & 0x3fu;

                    condition_2 = uint32_t(matrix_fragment_31) > uint32_t(matrix_accumulator_16);

                    matrix_accumulator_21 =
                        matrix_accumulator_23 * matrix_accumulator_22 + matrix_accumulator_21;

                    condition_5 = uint32_t(matrix_fragment_31) > uint32_t(matrix_fragment_19);

                    matrix_fragment_6 = matrix_fragment_28 ^ matrix_fragment_9;

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        int64_t(int32_t(matrix_fragment_12)) *
                                                int64_t(int32_t(matrix_accumulator_3)) +
                                            0ull);

                    matrix_fragment_15 = !condition_0 ? matrix_fragment_27 : packed_half_10;

                    condition_1 = uint32_t(matrix_fragment_31) > uint32_t(matrix_accumulator_21);

                    packed_half_10 = matrix_accumulator_32 ^ matrix_fragment_9;

                    condition_3 = int32_t(matrix_fragment_6) >= int32_t(0u);

                    matrix_fragment_6 = 0u - matrix_fragment_15;

                    matrix_accumulator_17 = packed_half_0 + matrix_fragment_15;

                    if (!condition_2) {
                        matrix_accumulator_16 =
                            matrix_accumulator_16 + (0u - matrix_accumulator_23);

                        matrix_fragment_14 = matrix_fragment_14 + 0x1u;
                    }

                    if (!condition_5) {
                        matrix_fragment_19 = matrix_fragment_19 + (0u - matrix_accumulator_23);
                    }

                    matrix_fragment_15 = matrix_fragment_9 * matrix_fragment_6 + matrix_fragment_24;

                    matrix_fragment_6 = matrix_fragment_36 ^ matrix_fragment_9;

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        int64_t(int32_t(matrix_fragment_4)) *
                                                int64_t(int32_t(matrix_accumulator_17)) +
                                            combine_words(matrix_fragment_12, matrix_fragment_13));

                    condition_6 = uint32_t(matrix_accumulator_16) >= uint32_t(matrix_fragment_31);

                    if (!condition_1) {
                        matrix_accumulator_21 =
                            matrix_accumulator_21 + (0u - matrix_accumulator_23);
                    }

                    condition_4 = uint32_t(matrix_fragment_19) >= uint32_t(matrix_fragment_31);

                    if (!condition_5) {
                        matrix_fragment_18 = matrix_fragment_18 + 0x1u;
                    }

                    condition_5 = int32_t(matrix_fragment_6) >= int32_t(0u);

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        uint64_t(uint32_t(matrix_fragment_15)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_12, matrix_fragment_13));

                    condition_2 = uint32_t(matrix_accumulator_21) >= uint32_t(matrix_fragment_31);

                    if (!condition_1) {
                        matrix_accumulator_20 = matrix_accumulator_20 + 0x1u;
                    }

                    matrix_accumulator_16 = matrix_fragment_30 & 0x3fu;

                    if (condition_6) {
                        matrix_fragment_14 = matrix_fragment_14 + 0x1u;
                    }

                    condition_6 = int32_t(packed_half_10) >= int32_t(0u);

                    if (condition_4) {
                        matrix_fragment_18 = matrix_fragment_18 + 0x1u;
                    }

                    packed_half_10 = matrix_fragment_34 & 0x3fu;

                    matrix_fragment_6 = matrix_fragment_14;

                    const uint64_t byte_address_6 = scaled_word_address(
                        combine_words(uniform_parameter_word_8, uniform_parameter_word_9),
                        combine_words(matrix_fragment_12, matrix_fragment_13));
                    matrix_fragment_14 = uint32_t(byte_address_6);

                    if (condition_2) {
                        matrix_accumulator_20 = matrix_accumulator_20 + 0x1u;
                    }

                    if (!condition_3) {
                        matrix_fragment_6 = 0u - matrix_fragment_6;
                    }

                    matrix_fragment_15 = uint32_t(byte_address_6 >> 32);

                    matrix_fragment_12 = matrix_fragment_26 & 0x3fu;

                    matrix_accumulator_17 = !condition_0 ? matrix_fragment_27 : matrix_fragment_6;

                    if (!condition_6) {
                        matrix_fragment_18 = 0u - matrix_fragment_18;
                    }

                    if (!condition_5) {
                        matrix_accumulator_20 = 0u - matrix_accumulator_20;
                    }

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        int64_t(int32_t(matrix_fragment_12)) *
                                                int64_t(int32_t(matrix_accumulator_3)) +
                                            0ull);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15), 0u, memory_ranges,
                        error_counts);

                    matrix_fragment_6 = 0u - matrix_accumulator_17;

                    matrix_accumulator_17 = packed_half_0 + matrix_accumulator_17;

                    matrix_accumulator_23 = !condition_0 ? matrix_fragment_27 : matrix_fragment_18;

                    matrix_fragment_27 = !condition_0 ? matrix_fragment_27 : matrix_accumulator_20;

                    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                                        int64_t(int32_t(packed_half_10)) *
                                                int64_t(int32_t(matrix_accumulator_3)) +
                                            0ull);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_25 = packed_half_0 + matrix_accumulator_23;

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        int64_t(int32_t(matrix_fragment_4)) *
                                                int64_t(int32_t(matrix_accumulator_17)) +
                                            combine_words(matrix_fragment_12, matrix_fragment_13));

                    matrix_fragment_29 = packed_half_0 + matrix_fragment_27;

                    packed_half_10 = 0u - matrix_fragment_27;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    split_address_words(matrix_accumulator_16, matrix_accumulator_17,
                                        int64_t(int32_t(matrix_accumulator_16)) *
                                                int64_t(int32_t(matrix_accumulator_3)) +
                                            0ull);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_19 = matrix_fragment_9 * matrix_fragment_6 + matrix_fragment_28;

                    matrix_fragment_6 = 0u - matrix_accumulator_23;

                    split_address_words(
                        matrix_accumulator_20, matrix_accumulator_21,
                        int64_t(int32_t(matrix_fragment_4)) * int64_t(int32_t(matrix_fragment_29)) +
                            combine_words(matrix_accumulator_20, matrix_accumulator_21));

                    split_address_words(matrix_fragment_12, matrix_fragment_13,
                                        uint64_t(uint32_t(matrix_fragment_19)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_12, matrix_fragment_13));

                    split_address_words(
                        matrix_fragment_18, matrix_fragment_19,
                        int64_t(int32_t(matrix_fragment_4)) * int64_t(int32_t(matrix_fragment_25)) +
                            combine_words(matrix_accumulator_16, matrix_accumulator_17));

                    const uint64_t byte_address_7 = scaled_word_address(
                        combine_words(uniform_parameter_word_8, uniform_parameter_word_9),
                        combine_words(matrix_fragment_12, matrix_fragment_13));
                    matrix_accumulator_16 = uint32_t(byte_address_7);

                    matrix_accumulator_17 =
                        matrix_fragment_9 * matrix_fragment_6 + matrix_accumulator_32;

                    matrix_fragment_6 = matrix_fragment_5 + matrix_fragment_34;

                    matrix_accumulator_23 = matrix_fragment_9 * packed_half_10 + matrix_fragment_36;

                    split_address_words(matrix_fragment_18, matrix_fragment_19,
                                        uint64_t(uint32_t(matrix_accumulator_17)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_18, matrix_fragment_19));

                    matrix_accumulator_17 = uint32_t(byte_address_7 >> 32);

                    split_address_words(
                        matrix_accumulator_20, matrix_accumulator_21,
                        uint64_t(uint32_t(matrix_accumulator_23)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_20, matrix_accumulator_21));

                    const uint64_t byte_address_8 = scaled_word_address(
                        combine_words(uniform_parameter_word_8, uniform_parameter_word_9),
                        combine_words(matrix_fragment_18, matrix_fragment_19));
                    matrix_fragment_12 = uint32_t(byte_address_8);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_16, matrix_accumulator_17), 0u,
                        memory_ranges, error_counts);

                    const uint64_t byte_address_9 = scaled_word_address(
                        combine_words(uniform_parameter_word_8, uniform_parameter_word_9),
                        combine_words(matrix_accumulator_20, matrix_accumulator_21));
                    matrix_accumulator_22 = uint32_t(byte_address_9);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_13 = uint32_t(byte_address_8 >> 32);

                    matrix_accumulator_23 = uint32_t(byte_address_9 >> 32);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    condition_0 = int32_t(matrix_fragment_6) >= int32_t(matrix_fragment_11);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_16, matrix_accumulator_17) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13), 0u, memory_ranges,
                        error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_22, matrix_accumulator_23), 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0xcull, 0u,
                        memory_ranges, error_counts);
                } while (!condition_0);
            }
        }
    }
    packed_half_0 = 0u - matrix_accumulator_2 + matrix_fragment_9;

    matrix_fragment_6 = min(int32_t(matrix_fragment_7), int32_t(packed_half_0));

    condition_0 = int32_t(matrix_fragment_6) >= int32_t(0x1u);

    if (!condition_0) {
        return;
    }

    matrix_fragment_7 = matrix_fragment_7 * packed_half_0;

    matrix_fragment_19 = matrix_fragment_7 * 0x40u;

    condition_0 = int32_t(matrix_fragment_8) >= int32_t(matrix_fragment_19);

    if (condition_0) {
        return;
    }

    // Clear the final column-padding region only. Earlier tile/edge publication
    // and its synchronization stay here; see docs/BOUNDARY_KERNELS.md.
    const BoundaryVectorLayout padding_layout{
        combine_words(parameters[parameter_output_downsample_optional_low_18],
                      parameters[parameter_output_downsample_optional_high_19]),
        matrix_accumulator_3, // component stride, uint32 words
        matrix_fragment_4, // row stride, uint32 words
        matrix_accumulator_2}; // column origin, signed-word bits
    const uint32_t padding_width = packed_half_0;
    const uint32_t first_padding_item = matrix_fragment_8;
    const uint32_t padding_extent = matrix_fragment_19;
    const uint32_t block_thread_count = matrix_fragment_5;
    clear_boundary_rows<6>(padding_layout, padding_width, first_padding_item,
                           padding_extent, block_thread_count, [&](uint64_t address) {
        pipeline_write<CheckBounds>(address, 0u, memory_ranges, error_counts);
    });
}
extern "C" __global__ __launch_bounds__(256) void phoenix_window_attention_channels_256_downsample(
    KernelArguments parameters) {
    run_window_attention_channels_256_downsample<false>(parameters.words, nullptr, nullptr);
}
