// DLSS5 Phoenix optimized kernel: window_attention/channels_32/downsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#include "../../../common/boundary_clear.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
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
    parameter_output_downsample_optional_low_16 = 16,
    parameter_output_downsample_optional_high_17 = 17,
    parameter_tensor_height_18 = 18,
    parameter_tensor_width_19 = 19,
};

struct alignas(8) KernelArguments {
    uint32_t words[24];
};
static_assert(sizeof(KernelArguments) == 96, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_32_downsample(
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
    uint32_t matrix_fragment_44 = 0;
    uint32_t matrix_fragment_45 = 0;
    uint32_t packed_half_46 = 0;
    uint32_t matrix_fragment_47 = 0;
    uint32_t matrix_fragment_48 = 0;
    uint32_t matrix_fragment_49 = 0;
    uint32_t packed_half_50 = 0;
    uint32_t packed_half_51 = 0;
    uint32_t matrix_fragment_52 = 0;
    uint32_t matrix_fragment_53 = 0;
    uint32_t matrix_fragment_54 = 0;
    uint32_t matrix_fragment_55 = 0;
    uint32_t matrix_fragment_56 = 0;
    uint32_t matrix_fragment_57 = 0;
    uint32_t matrix_fragment_58 = 0;
    uint32_t matrix_fragment_59 = 0;
    uint32_t packed_half_60 = 0;
    uint32_t packed_half_61 = 0;
    uint32_t packed_half_62 = 0;
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
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
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
    uint32_t matrix_accumulator_96 = 0;
    uint32_t matrix_accumulator_97 = 0;
    uint32_t matrix_fragment_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t packed_half_100 = 0;
    uint32_t packed_half_101 = 0;
    uint32_t matrix_accumulator_102 = 0;
    uint32_t packed_half_103 = 0;
    uint32_t packed_half_104 = 0;
    uint32_t packed_half_105 = 0;
    uint32_t packed_half_106 = 0;
    uint32_t packed_half_107 = 0;
    uint32_t packed_half_108 = 0;
    uint32_t packed_half_109 = 0;
    uint32_t packed_half_110 = 0;
    uint32_t packed_half_111 = 0;
    uint32_t packed_half_112 = 0;
    uint32_t packed_half_113 = 0;
    uint32_t packed_half_114 = 0;
    uint32_t matrix_accumulator_115 = 0;
    uint32_t matrix_accumulator_116 = 0;
    uint32_t matrix_accumulator_117 = 0;
    uint32_t matrix_accumulator_118 = 0;
    uint32_t matrix_accumulator_119 = 0;
    uint32_t matrix_accumulator_120 = 0;
    uint32_t matrix_accumulator_121 = 0;
    uint32_t packed_half_122 = 0;
    uint32_t packed_half_123 = 0;
    uint32_t matrix_fragment_124 = 0;
    uint32_t matrix_fragment_125 = 0;
    uint32_t matrix_accumulator_126 = 0;
    uint32_t matrix_accumulator_127 = 0;
    uint32_t matrix_accumulator_128 = 0;
    uint32_t matrix_accumulator_129 = 0;
    uint32_t matrix_accumulator_130 = 0;
    uint32_t matrix_accumulator_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t matrix_accumulator_133 = 0;
    uint32_t packed_half_134 = 0;
    uint32_t packed_half_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t address_word_138 = 0;
    uint32_t address_word_139 = 0;
    uint32_t packed_half_140 = 0;
    uint32_t packed_half_141 = 0;
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
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t address_word_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t packed_half_163 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t packed_half_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_parameter_word_15 = 0;
    uint32_t uniform_reused_word_16 = 0;
    uint32_t uniform_reused_word_17 = 0;
    uint32_t uniform_block_index_18 = 0;
    uint32_t uniform_index_word_19 = 0;
    uint32_t uniform_index_word_20 = 0;
    uint32_t uniform_address_word_21 = 0;
    uint32_t uniform_index_word_22 = 0;
    uint32_t uniform_index_word_23 = 0;
    uint32_t uniform_index_word_24 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
    bool condition_5 = false;
    bool condition_6 = false;
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

    uniform_index_word_13 = parameters[parameter_optional_control_13];

    uniform_index_word_7 = blockIdx.y;

    packed_half_0 = threadIdx.x;

    reused_word_1 = reused_word_1 + (0u - 0x8u);

    uniform_index_word_8 = parameters[parameter_tensor_height_6];

    matrix_accumulator_3 = threadIdx.y;

    uniform_index_word_10 = parameters[parameter_optional_control_8];
    uniform_index_word_11 = parameters[parameter_scalar_control_9];

    uniform_index_word_6 = blockIdx.x;

    matrix_accumulator_2 = threadIdx.z;

    uniform_index_word_12 = parameters[parameter_scalar_control_12];

    uniform_address_word_9 = parameters[parameter_tensor_width_7];

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_index_word_13));

    uniform_reused_word_16 = 0u;
    uniform_reused_word_17 = 0u;

    uniform_index_word_8 = 0u - uniform_index_word_4 + uniform_index_word_8;

    uniform_index_word_7 = (uniform_index_word_7 << 0x3u) + uniform_index_word_11;

    uniform_index_word_4 = 0u;

    uniform_condition_2 = uint32_t(uniform_index_word_8) != 0u;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_12));

    uniform_index_word_6 = (uniform_index_word_6 << 0x3u) + uniform_index_word_10;

    uniform_index_word_10 = uniform_index_word_7 + (0u - uniform_index_word_13);

    uniform_index_word_5 = 0u - uniform_index_word_5 + uniform_address_word_9;

    uniform_index_word_12 = uniform_index_word_6 + (0u - uniform_index_word_12);

    uniform_address_word_14 = uniform_index_word_10 + 0x8u;

    uniform_index_word_11 = uniform_index_word_10 + 0x7u;

    if (uniform_condition_2) {
        uniform_index_word_8 = uniform_index_word_8 + 0xffffffffu;
    }

    uniform_condition_1 = uint32_t(uniform_index_word_5) != 0u;

    uniform_condition_0 = int32_t(uniform_address_word_14) > int32_t(0u);

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_2) {
        uniform_address_word_9 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);
    }

    condition_0 = (matrix_accumulator_2 | matrix_accumulator_3 | packed_half_0) != 0;

    uniform_parameter_word_15 = uniform_index_word_12 + 0x8u;

    packed_half_0 = max(int32_t(0u), int32_t(uniform_index_word_10));

    if (uniform_condition_2) {
        uniform_address_word_9 = uint32_t(combine_words(uniform_address_word_9, 0u) << 0x3u >> 32) +
                                 uniform_index_word_8;
    }

    uniform_index_word_8 = uniform_condition_0 ? uniform_index_word_11 : 0u;

    uniform_index_word_13 = uniform_index_word_12 + 0x7u;

    uniform_condition_0 = int32_t(uniform_parameter_word_15) > int32_t(0u);

    if (uniform_condition_2) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_address_word_9)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    packed_half_0 = uint32_t(uint32_t(packed_half_0) >> 0x3u);

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_5 + 0xffffffffu;
    }

    uniform_index_word_13 =
        uint32_t(combine_words(uniform_index_word_13, 0u) << 0x1du >> 32) + 0x1u;

    uniform_index_word_8 = min(int32_t(uniform_index_word_8), int32_t(uniform_index_word_4));

    matrix_accumulator_2 = max(int32_t(0u), int32_t(uniform_index_word_12));

    if (uniform_condition_1) {
        uniform_address_word_9 = uint32_t(int32_t(uniform_index_word_5) >> 0x1fu);
    }

    uniform_index_word_4 = 0u;

    condition_0 = int32_t(packed_half_0) >= int32_t(uniform_index_word_8) || condition_0;

    if (uniform_condition_1) {
        uniform_index_word_5 = uint32_t(combine_words(uniform_address_word_9, 0u) << 0x3u >> 32) +
                               uniform_index_word_5;
    }

    uniform_address_word_9 = uniform_condition_0 ? uniform_index_word_13 : 0u;

    if (uniform_condition_1) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_5)) * (1ll << 0x1du) >> 32) + 0x1u;
    }
    if (!condition_0) {
        matrix_fragment_6 = uniform_index_word_4;

        matrix_fragment_4 = uint32_t(uint32_t(matrix_accumulator_2) >> 0x3u);

        matrix_fragment_6 = min(int32_t(matrix_fragment_6), int32_t(uniform_address_word_9));
        do {
            condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_6);
            if (!condition_0) {
                matrix_fragment_8 = packed_half_0 * uniform_index_word_4;

                matrix_fragment_5 = matrix_fragment_4;
                do {
                    matrix_accumulator_2 = parameters[parameter_wait_optional_low_10];
                    matrix_accumulator_3 = parameters[parameter_wait_optional_high_11];

                    matrix_fragment_7 = matrix_fragment_8 + matrix_fragment_5;

                    split_address_words(
                        matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));

                    matrix_fragment_7 = pipeline_read<CheckBounds>(
                        combine_words(matrix_accumulator_2, matrix_accumulator_3), memory_ranges,
                        error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_6);

                    condition_0 = int32_t(matrix_fragment_7) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_7 = pipeline_read<CheckBounds>(
                                combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                memory_ranges, error_counts);

                            condition_0 = int32_t(matrix_fragment_7) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            packed_half_0 = packed_half_0 + 0x1u;

            condition_0 = uint32_t(packed_half_0) != uint32_t(uniform_index_word_8);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    uniform_address_word_14 = parameters[parameter_tensor_height_6];
    uniform_parameter_word_15 = parameters[parameter_tensor_width_7];

    matrix_fragment_13 = threadIdx.x & 31u;

    address_word_138 = parameters[parameter_trained_weights_low_4];
    address_word_139 = parameters[parameter_trained_weights_high_5];

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_7) >> 0x1fu);

    uniform_condition_3 = false;

    uniform_condition_2 = true;

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_6;

    uniform_index_word_10 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_7;

    uniform_condition_6 = false;

    address_word_158 = parameters[parameter_trained_weights_low_4];
    address_word_159 = parameters[parameter_trained_weights_high_5];

    uniform_index_word_11 = uint32_t(int32_t(uniform_address_word_14) >> 0x1fu);

    uniform_index_word_13 = uniform_address_word_14 & 0xfffffffcu;

    uniform_index_word_12 = uint32_t(int32_t(uniform_parameter_word_15) >> 0x1fu);

    uniform_index_word_24 = uniform_parameter_word_15 & 0xfffffffcu;

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x2u >> 32) + uniform_address_word_14;

    uniform_condition_1 = uint32_t(uniform_index_word_13) != uint32_t(0x4u);

    uniform_address_word_14 = uint32_t(combine_words(uniform_index_word_12, 0u) << 0x2u >> 32) +
                              uniform_parameter_word_15;

    uniform_index_word_13 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_index_word_12 = uint32_t(int32_t(uniform_index_word_10) >> 0x2u);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    uniform_block_index_18 = uint32_t(int32_t(uniform_address_word_14) >> 0x2u);

    uniform_index_word_20 = uniform_index_word_13 + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_13) >= int32_t(uniform_block_index_18);

    split_address_words(matrix_accumulator_64, matrix_accumulator_65,
                        int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_138, address_word_139));

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_12 + 0x1u;

        uniform_condition_4 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_11);
    }

    uniform_condition_0 =
        int32_t(uniform_index_word_6) > int32_t(0u - 0x4u) && !uniform_condition_0;

    __syncthreads();

    if (uniform_condition_1) {
        uniform_condition_5 = int32_t(uniform_index_word_5) >= int32_t(uniform_index_word_11);

        uniform_condition_3 =
            int32_t(uniform_index_word_7) < int32_t(0u - 0x3u) || uniform_condition_4;
    }

    uniform_index_word_22 = uint32_t(uniform_condition_0);

    uniform_condition_4 = false;

    if (uniform_condition_1) {
        uniform_condition_4 =
            int32_t(uniform_index_word_7) < int32_t(0u - 0x7u) || uniform_condition_5;

        uniform_condition_2 = !uniform_condition_3;
    }

    uniform_condition_5 = int32_t(uniform_index_word_22) != int32_t(0u);

    uniform_condition_0 = uint32_t(uniform_index_word_24) == uint32_t(0x4u) || uniform_condition_3;

    uniform_address_word_14 = uniform_condition_3 ? uniform_index_word_13 : 0u;

    uniform_condition_3 = true;

    if (uniform_condition_1) {
        uniform_condition_3 = !uniform_condition_4;
    }

    uniform_condition_2 = uniform_condition_2 & (uniform_condition_0 | uniform_condition_5);

    uniform_condition_0 = uint32_t(uniform_index_word_24) == uint32_t(0x4u) || uniform_condition_4;

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_12 + 0x1u;
    }

    uniform_address_word_21 = uniform_condition_4 ? uniform_index_word_13 : 0u;

    if (uniform_condition_1) {
        uniform_condition_4 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_11);
    }

    uniform_condition_3 = uniform_condition_3 & (uniform_condition_0 | uniform_condition_5);

    if (uniform_condition_1) {
        uniform_condition_0 = int32_t(uniform_index_word_5) >= int32_t(uniform_index_word_11);
    }

    uniform_condition_5 = true;

    if (uniform_condition_1) {
        uniform_condition_6 =
            int32_t(uniform_index_word_7) < int32_t(0u - 0x3u) || uniform_condition_4;
    }

    uniform_condition_4 = false;

    if (uniform_condition_1) {
        uniform_condition_4 =
            int32_t(uniform_index_word_7) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    uniform_condition_0 = int32_t(uniform_index_word_20) >= int32_t(uniform_block_index_18);

    uniform_index_word_23 = uniform_condition_6 ? 0x40u : 0u;

    if (uniform_condition_2) {
        uniform_index_word_10 = parameters[parameter_input_low_0];
        uniform_index_word_11 = parameters[parameter_input_high_1];
    }

    condition_0 = uniform_condition_2;

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_64, matrix_accumulator_65),
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    condition_1 = uniform_condition_3;

    uniform_condition_0 =
        int32_t(uniform_index_word_6) > int32_t(0u - 0x8u) && !uniform_condition_0;

    if (uniform_condition_1) {
        uniform_condition_5 = !uniform_condition_4;
    }

    uniform_condition_6 = uint32_t(uniform_index_word_24) == uint32_t(0x4u) || uniform_condition_4;

    uniform_index_word_5 = 0u;

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_12 * uniform_block_index_18;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x200ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    uniform_condition_5 = uniform_condition_5 & (uniform_condition_6 | uniform_condition_0);

    uniform_condition_6 = uint32_t(uniform_index_word_24) != uint32_t(0x4u);

    packed_half_0 = matrix_fragment_13;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1000ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    uniform_address_word_14 =
        !uniform_condition_6 ? uniform_address_word_14 : uniform_index_word_13;

    uniform_address_word_21 =
        !uniform_condition_6 ? uniform_address_word_21 : uniform_index_word_13;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1200ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    if (uniform_condition_2) {
        uniform_index_word_5 = uniform_address_word_14 + uniform_index_word_5;
    }

    if (!uniform_condition_2) {
        uniform_address_word_14 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_2) {
        uniform_index_word_5 = uniform_index_word_5 << (0x7u & 31u);
    }

    if (!uniform_condition_2) {
        uniform_index_word_19 =
            permute_signed(uniform_address_word_14, uniform_address_word_14, 0x5410u);
    }

    if (uniform_condition_2) {
        split_address_words(uniform_index_word_10, uniform_index_word_11,
                            int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_index_word_10, uniform_index_word_11));
    }

    if (!condition_0) {
        matrix_fragment_16 = uniform_index_word_19;

        matrix_fragment_17 = uniform_index_word_19;

        matrix_fragment_18 = uniform_index_word_19;
    }

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_index_word_10, uniform_index_word_11));

    if (!condition_0) {
        matrix_fragment_19 = uniform_index_word_19;
    }

    if (uniform_condition_3) {
        uniform_address_word_14 = parameters[parameter_input_low_0];
        uniform_parameter_word_15 = parameters[parameter_input_high_1];
    }

    if (condition_0) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_1) {
        uniform_index_word_5 =
            uniform_index_word_12 * uniform_block_index_18 + uniform_block_index_18;
    }

    packed_half_0 = packed_half_0 & 0xfffffffcu;

    if (uniform_condition_5) {
        uniform_index_word_10 = parameters[parameter_input_low_0];
        uniform_index_word_11 = parameters[parameter_input_high_1];
    }

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_address_word_21 + uniform_index_word_5;
    }

    if (!uniform_condition_3) {
        uniform_address_word_21 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_5 << (0x7u & 31u);
    }

    if (!uniform_condition_3) {
        uniform_address_word_21 =
            permute_signed(uniform_address_word_21, uniform_address_word_21, 0x5410u);
    }

    if (uniform_condition_3) {
        split_address_words(uniform_address_word_14, uniform_parameter_word_15,
                            int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_address_word_14, uniform_parameter_word_15));
    }

    packed_half_161 = 0u - packed_half_0 + matrix_fragment_13;

    uniform_condition_3 = int32_t(uniform_index_word_23) != int32_t(0u);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(uniform_address_word_14, uniform_parameter_word_15));

    if (!condition_1) {
        matrix_fragment_4 = uniform_address_word_21;

        matrix_fragment_5 = uniform_address_word_21;

        matrix_fragment_6 = uniform_address_word_21;

        matrix_fragment_7 = uniform_address_word_21;
    }

    if (condition_1) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    split_address_words(packed_half_160, packed_half_161,
                        int64_t(int32_t(packed_half_161)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_138, address_word_139));

    uniform_address_word_14 = uniform_condition_3 ? uniform_index_word_20 : 0u;

    uniform_index_word_5 = 0u;

    if (uniform_condition_1) {
        uniform_index_word_5 =
            uniform_index_word_12 * uniform_block_index_18 + uniform_block_index_18;
    }

    matrix_fragment_45 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, packed_half_161) + 0x2010ull, memory_ranges, error_counts);

    uniform_address_word_14 =
        !uniform_condition_6 ? uniform_address_word_14 : uniform_index_word_20;

    if (!uniform_condition_6) {
        uniform_index_word_20 = uniform_condition_4 ? uniform_index_word_20 : 0u;
    }

    packed_half_163 = matrix_fragment_13 + 0x4u + (0u - packed_half_0);

    if (uniform_condition_5) {
        uniform_index_word_5 = uniform_index_word_20 + uniform_index_word_5;
    }

    split_address_words(packed_half_162, packed_half_163,
                        uint64_t(uint32_t(packed_half_163)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_138, address_word_139));

    condition_0 = uniform_condition_5;

    if (uniform_condition_5) {
        uniform_index_word_5 = uniform_index_word_5 << (0x7u & 31u);
    }

    matrix_accumulator_143 = matrix_fragment_13 + 0x8u + (0u - packed_half_0);

    if (!uniform_condition_5) {
        uniform_parameter_word_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                                  select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_44 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_162, packed_half_163) + 0x2010ull, memory_ranges, error_counts);

    matrix_accumulator_3 = matrix_fragment_13 + 0xcu + (0u - packed_half_0);

    if (uniform_condition_5) {
        split_address_words(uniform_index_word_10, uniform_index_word_11,
                            int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_index_word_10, uniform_index_word_11));
    }

    split_address_words(matrix_accumulator_142, matrix_accumulator_143,
                        uint64_t(uint32_t(matrix_accumulator_143)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_138, address_word_139));

    if (!uniform_condition_5) {
        uniform_parameter_word_15 =
            permute_signed(uniform_parameter_word_15, uniform_parameter_word_15, 0x5410u);
    }

    split_address_words(address_word_138, address_word_139,
                        uint64_t(uint32_t(matrix_accumulator_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_138, address_word_139));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_index_word_10, uniform_index_word_11));

    uniform_condition_2 = true;

    if (!condition_0) {
        matrix_fragment_8 = uniform_parameter_word_15;

        matrix_fragment_9 = uniform_parameter_word_15;

        matrix_fragment_10 = uniform_parameter_word_15;

        matrix_fragment_11 = uniform_parameter_word_15;
    }

    if (condition_0) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));
    }

    if (uniform_condition_1) {
        uniform_condition_2 = !uniform_condition_3;
    }

    uniform_condition_3 = uint32_t(uniform_index_word_24) == uint32_t(0x4u) || uniform_condition_3;

    matrix_fragment_49 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x2010ull, memory_ranges,
        error_counts);

    uniform_condition_2 = uniform_condition_2 & (uniform_condition_3 | uniform_condition_0);

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    matrix_fragment_48 = load_aligned_word<CheckBounds>(
        combine_words(address_word_138, address_word_139) + 0x2010ull, memory_ranges, error_counts);

    uniform_index_word_5 = 0u;

    if (uniform_condition_2) {
        uniform_index_word_10 = parameters[parameter_input_low_0];
        uniform_index_word_11 = parameters[parameter_input_high_1];
    }

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_12 * uniform_block_index_18;
    }

    condition_0 = uniform_condition_2;

    uniform_block_index_18 = parameters[parameter_scalar_control_9];

    if (uniform_condition_2) {
        uniform_index_word_5 = uniform_address_word_14 + uniform_index_word_5;
    }

    packed_half_0 = 0x2b28u;

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    if (uniform_condition_2) {
        uniform_index_word_5 = uniform_index_word_5 << (0x7u & 31u);
    }

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    if (uniform_condition_2) {
        split_address_words(uniform_index_word_10, uniform_index_word_11,
                            int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_index_word_10, uniform_index_word_11));
    }

    if (!uniform_condition_2) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    split_address_words(matrix_fragment_20, matrix_fragment_21,
                        combine_words(uniform_index_word_10, uniform_index_word_11));

    if (!uniform_condition_2) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    packed_half_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

    uniform_index_word_10 = parameters[parameter_tensor_height_6];
    uniform_index_word_11 = parameters[parameter_tensor_width_7];

    if (condition_0) {
        split_address_words(matrix_fragment_20, matrix_fragment_21,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_20, matrix_fragment_21));
    }

    if (!condition_0) {
        matrix_fragment_12 = uniform_index_word_5;

        matrix_fragment_13 = uniform_index_word_5;

        matrix_fragment_14 = uniform_index_word_5;

        matrix_fragment_15 = uniform_index_word_5;
    }

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_20, matrix_fragment_21), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    uniform_condition_6 = int32_t(uniform_index_word_22) != int32_t(0u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    uniform_address_word_14 = parameters[parameter_output_low_2];
    uniform_parameter_word_15 = parameters[parameter_output_high_3];

    packed_half_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_50), 0xc400c400u);

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_fragment_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_21),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_21),
        select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_53),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_51),
        select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    packed_half_50 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                         select_half_operand<2, false, false>(packed_half_50));

    matrix_fragment_22 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_23 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

    packed_half_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_22),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_22),
        select_half_operand<2, false, false>(matrix_fragment_23), 0x3b283b28u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_51),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(packed_half_46));

    packed_half_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x44004400u);

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_50),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0xc400c400u);

    packed_half_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

    packed_half_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_46),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0xc400c400u);

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_46),
        select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_53),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_50),
        select_half_operand<2, false, false>(packed_half_51), 0x3b283b28u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(packed_half_46), 0x3b283b28u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_51));

    packed_half_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(packed_half_46));

    packed_half_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0xc400c400u);

    packed_half_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    packed_half_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    packed_half_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    packed_half_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_60),
        select_half_operand<2, false, false>(packed_half_61), 0x3b283b28u);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_50),
        select_half_operand<2, false, false>(packed_half_51), 0x3b283b28u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    packed_half_61 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                         select_half_operand<2, false, false>(packed_half_61));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_51));

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    multiply_accumulate_half(packed_half_50, packed_half_51, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_42,
                             matrix_fragment_43, 0u, 0u);

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x600ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_6);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_7);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_19);

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_22 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_78,
                             matrix_accumulator_79);

    packed_half_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_50), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_22,
                             matrix_fragment_23);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x400ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_2,
                             matrix_accumulator_3);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    packed_half_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    packed_half_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    packed_half_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_60),
        select_half_operand<2, false, false>(packed_half_61), 0x3b283b28u);

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_50),
                            select_half_operand<2, false, false>(matrix_accumulator_63));

    packed_half_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    packed_half_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    packed_half_61 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                         select_half_operand<2, false, false>(packed_half_61));

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_50), 0xc400c400u);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    packed_half_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_51),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(packed_half_60), 0x3b283b28u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_50),
        select_half_operand<2, false, false>(packed_half_51), 0x3b283b28u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    packed_half_60 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                         select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_51));

    multiply_accumulate_half(packed_half_50, packed_half_51, matrix_fragment_12, matrix_fragment_13,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_38,
                             matrix_fragment_39, 0u, 0u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_52 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_53 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_60),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_61),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_52 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_53 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_52 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_53 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_52 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_53 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_accumulator_96 = unpack_fp8_bits(matrix_fragment_12);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1400ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    matrix_accumulator_97 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    packed_half_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_50), 0x44004400u);

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0x44004400u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    packed_half_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0xc400c400u);

    matrix_accumulator_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0xc400c400u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_61),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_63),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_61),
        select_half_operand<2, false, false>(packed_half_62), 0x3b283b28u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(packed_half_60), 0x3b283b28u);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_50),
                                         select_half_operand<2, false, false>(packed_half_62));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_51),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    packed_half_60 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                         select_half_operand<2, false, false>(packed_half_60));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_60),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x800ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_96,
                             matrix_accumulator_97);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_44));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1600ull, matrix_fragment_44,
        matrix_fragment_45, packed_half_46, matrix_fragment_47, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0xa00ull, matrix_fragment_48,
        matrix_fragment_49, packed_half_50, packed_half_51, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_12, matrix_fragment_13,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_22,
                             matrix_fragment_23, 0u, 0u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95), 0x44004400u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94), 0x44004400u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_99),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<2, false, false>(packed_half_100), 0x3b283b28u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x44004400u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    matrix_fragment_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0xc400c400u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0xc400c400u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_108),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<2, false, false>(packed_half_109), 0x3b283b28u);

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_99 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                             select_half_operand<2, false, false>(packed_half_106));

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_104));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

    packed_half_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                                         select_half_operand<2, false, false>(packed_half_101));

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                                          select_half_operand<2, false, false>(packed_half_107));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_26,
                             matrix_fragment_27, 0u, 0u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                          select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_25),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_23 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_20),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0xc400c400u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_22),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_108),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_22),
        select_half_operand<2, false, false>(matrix_fragment_23), 0x3b283b28u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x44004400u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x44004400u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0xc400c400u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_20),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_23 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_accumulator_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_115),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0xc400c400u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115),
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3b283b28u);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_22),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    packed_half_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                   select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_25),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(0u));

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_22),
        select_half_operand<2, false, false>(matrix_fragment_23), 0x3b283b28u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                          select_half_operand<2, false, false>(packed_half_114));

    packed_half_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                   select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_116 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                      select_half_operand<2, false, false>(0u));

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(packed_half_109));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1800ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x44004400u);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                      select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_30),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1a00ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_25),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_68,
                             matrix_fragment_69);

    packed_half_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_31));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0xe00ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_82,
                             matrix_accumulator_83);

    packed_half_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                   select_half_operand<2, false, false>(packed_half_111));

    matrix_accumulator_63 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(matrix_fragment_69));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0xc00ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, packed_half_60,
                             packed_half_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, packed_half_60,
                             packed_half_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, packed_half_60,
                             packed_half_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, packed_half_60, packed_half_61,
                             packed_half_62, matrix_accumulator_63, packed_half_46,
                             matrix_fragment_47, matrix_fragment_92, matrix_fragment_93);

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_16, matrix_fragment_17,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                             matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_46, matrix_fragment_47, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x44004400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    packed_half_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x44004400u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    packed_half_106 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_50, packed_half_51, 0u, 0u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                          select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0xc400c400u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                             select_half_operand<2, false, false>(packed_half_111));

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(packed_half_46), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             packed_half_50, packed_half_51, 0u, 0u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(packed_half_101));

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    packed_half_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_44),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                          select_half_operand<2, false, false>(packed_half_114));

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0xc400c400u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_45),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                          select_half_operand<2, false, false>(packed_half_100));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_110));

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_50, packed_half_51,
                             0u, 0u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_46),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                                          select_half_operand<2, false, false>(packed_half_106));

    packed_half_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_107));

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_50, packed_half_51, 0u, 0u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    packed_half_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_46),
        select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

    packed_half_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_44),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0xc400c400u);

    packed_half_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_46),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_46),
        select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    packed_half_50 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                         select_half_operand<2, false, false>(packed_half_50));

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_47),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_50),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x44004400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

    matrix_accumulator_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                          select_half_operand<2, false, false>(packed_half_113));

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_118),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(matrix_fragment_36));

    packed_half_51 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                         select_half_operand<2, false, false>(packed_half_51));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                          select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(packed_half_109));

    packed_half_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x44004400u);

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3b283b28u);

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_118),
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3b283b28u);

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    matrix_fragment_38 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                            select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_accumulator_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0xc400c400u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1c00ull, matrix_fragment_44,
        matrix_fragment_45, packed_half_46, matrix_fragment_47, memory_ranges, error_counts);

    packed_half_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1e00ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_116),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_61),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116),
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3b283b28u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_71),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_61),
        select_half_operand<2, false, false>(matrix_fragment_70), 0x3b283b28u);

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<2, false, false>(packed_half_100), 0x3b283b28u);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(packed_half_60), 0x3b283b28u);

    packed_half_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    packed_half_60 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                         select_half_operand<2, false, false>(packed_half_60));

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(0u));

    packed_half_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                   select_half_operand<2, false, false>(0u));

    packed_half_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_51),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    packed_half_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                   select_half_operand<2, false, false>(0u));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_60),
                                       select_half_operand<2, false, false>(packed_half_61));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(packed_half_51));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(packed_half_50));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_86,
                             matrix_accumulator_87);

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    packed_half_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                   select_half_operand<2, false, false>(matrix_accumulator_102));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_76,
                             matrix_accumulator_77);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2260ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    packed_half_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                   select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_102, packed_half_103, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_92,
                             matrix_fragment_93);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2060ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_98,
                             matrix_fragment_99);

    packed_half_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                   select_half_operand<2, false, false>(packed_half_50));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_48,
                             matrix_fragment_49, packed_half_50, packed_half_51, matrix_fragment_32,
                             matrix_fragment_33, matrix_accumulator_66, matrix_accumulator_67);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_48,
                             matrix_fragment_49, packed_half_50, packed_half_51, matrix_fragment_34,
                             matrix_fragment_35, matrix_accumulator_2, matrix_accumulator_3);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_12, matrix_fragment_13,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_26,
                             matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_12 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_126), 0x44004400u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_fragment_13 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_127), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_48,
                             matrix_fragment_49, packed_half_50, packed_half_51, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_94, matrix_fragment_95);

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_48,
                             matrix_fragment_49, packed_half_50, packed_half_51, matrix_fragment_22,
                             matrix_fragment_23, matrix_accumulator_82, matrix_accumulator_83);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2460ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_12),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_14),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

    matrix_fragment_15 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_12),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_14),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_15),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_15),
        select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2660ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    multiply_accumulate_half(packed_half_50, packed_half_51, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_24,
                             matrix_fragment_25, 0u, 0u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x44004400u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x44004400u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_accumulator_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0xc400c400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_fragment_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

    matrix_accumulator_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_50), 0x44004400u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_6),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                            select_half_operand<2, false, false>(matrix_accumulator_88));

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_71),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0x44004400u);

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_6),
        select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

    matrix_fragment_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_66),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3b283b28u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3b283b28u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_accumulator_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0xc400c400u);

    matrix_accumulator_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_106),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_66),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_67),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    matrix_fragment_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

    matrix_fragment_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_accumulator_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x44004400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    matrix_accumulator_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0xc400c400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    matrix_accumulator_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_8),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

    packed_half_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                         select_half_operand<2, false, false>(packed_half_100));

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0xc400c400u);

    packed_half_51 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_51),
                                         select_half_operand<2, false, false>(packed_half_101));

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x44004400u);

    matrix_fragment_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_10),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89), 0xc400c400u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_66),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0xc400c400u);

    packed_half_61 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                                         select_half_operand<2, false, false>(packed_half_104));

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    matrix_fragment_11 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_10),
        select_half_operand<2, false, false>(matrix_fragment_11), 0x3b283b28u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_fragment_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_67),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_accumulator_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0xc400c400u);

    matrix_fragment_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<2, false, false>(matrix_fragment_10), 0x3b283b28u);

    matrix_accumulator_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    packed_half_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_50),
                                         select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_fragment_71), 0x3b283b28u);

    matrix_fragment_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0xc400c400u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_66),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_61),
                                       select_half_operand<2, false, false>(0u));

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3b283b28u);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                          select_half_operand<2, false, false>(packed_half_105));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_4),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_125 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_60),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_51),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_50),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    matrix_fragment_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_accumulator_88));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_70),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_18 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_16),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0xc400c400u);

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_18),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_16),
        select_half_operand<2, false, false>(matrix_fragment_17), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2860ull, packed_half_60,
        packed_half_61, packed_half_62, matrix_accumulator_63, memory_ranges, error_counts);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_4),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_18),
        select_half_operand<2, false, false>(matrix_fragment_19), 0x3b283b28u);

    matrix_fragment_124 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3b283b28u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(matrix_accumulator_89), 0x3b283b28u);

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_124,
                             matrix_fragment_125, matrix_accumulator_126, matrix_accumulator_127,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                       select_half_operand<2, false, false>(matrix_fragment_24));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2a60ull,
        matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
        memory_ranges, error_counts);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                      select_half_operand<2, false, false>(matrix_fragment_19));

    multiply_accumulate_half(packed_half_114, matrix_accumulator_115, matrix_fragment_124,
                             matrix_fragment_125, matrix_accumulator_126, matrix_accumulator_127,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_124,
                             matrix_fragment_125, matrix_accumulator_126, matrix_accumulator_127,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_124, matrix_fragment_125, matrix_fragment_124,
                             matrix_fragment_125, matrix_accumulator_126, matrix_accumulator_127,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_74,
                             matrix_fragment_75);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_92,
                             matrix_fragment_93);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_94,
                             matrix_fragment_95);

    multiply_accumulate_half(packed_half_50, packed_half_51, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_44,
                             matrix_fragment_45, matrix_accumulator_86, matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_102, packed_half_103, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_102,
                             packed_half_103);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_2,
                             matrix_accumulator_3);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_125),
                                         select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_accumulator_3 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                         select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_50),
                                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_2 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_3 = parameters[parameter_trained_weights_high_5];

    matrix_fragment_11 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x4c60ull, memory_ranges,
        error_counts);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_124),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_93 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                       select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_134, packed_half_135, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_54,
                             matrix_fragment_55, 0u, 0u);

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_51),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(matrix_accumulator_128));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                      select_half_operand<2, false, false>(packed_half_104),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                      select_half_operand<2, false, false>(packed_half_105),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_accumulator_132),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(packed_half_140, packed_half_141, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_fragment_20 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                      select_half_operand<2, false, false>(packed_half_134),
                                      select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
    }

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    packed_half_0 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_accumulator_129));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_accumulator_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                      select_half_operand<2, false, false>(packed_half_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(matrix_accumulator_133),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                                   select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    multiply_accumulate_half(packed_half_122, packed_half_123, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_12,
                             matrix_fragment_13, 0u, 0u);

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    multiply_accumulate_half(packed_half_150, packed_half_151, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_14,
                             matrix_fragment_15, 0u, 0u);

    matrix_fragment_20 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(packed_half_148, packed_half_149, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_22,
                             matrix_fragment_23, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    multiply_accumulate_half(packed_half_144, packed_half_145, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                                      select_half_operand<2, false, false>(matrix_accumulator_76),
                                      select_half_operand<2, false, false>(matrix_fragment_22));

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_29 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(matrix_accumulator_79));

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_29), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_30 = result_word;
    }

    matrix_fragment_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                                      select_half_operand<2, false, false>(matrix_accumulator_77),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(packed_half_146, packed_half_147, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_13 = permute_signed(matrix_accumulator_2, matrix_accumulator_2, 0x5432u);

    matrix_fragment_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                             select_half_operand<2, false, false>(packed_half_122));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = permute_signed(matrix_accumulator_3, matrix_accumulator_3, 0x5432u);

    matrix_fragment_20 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x4100410u);

    matrix_fragment_74 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                             select_half_operand<2, false, false>(packed_half_150));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_74 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                      select_half_operand<2, false, false>(packed_half_148),
                                      select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                             select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_14 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_23 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    matrix_fragment_22 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                      select_half_operand<2, false, false>(packed_half_149),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                             select_half_operand<2, false, false>(packed_half_123));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_20 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                             select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_21 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                             select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_21 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(matrix_fragment_21));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_82 = result_word;
    }

    matrix_fragment_14 = rsqrt_bits(matrix_fragment_14);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_22 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_31 = rsqrt_bits(matrix_fragment_23);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_84 = result_word;
    }

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_22), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_83 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    matrix_fragment_28 = half_to_float_bits((matrix_fragment_8 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_29 = result_word;
    }

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_8 >> 16) * 65537u);

    matrix_fragment_74 = rsqrt_bits(matrix_fragment_28);

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_13 = pack_half(matrix_fragment_31, matrix_fragment_14);

    matrix_fragment_14 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                             select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                      select_half_operand<2, false, false>(packed_half_144),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_28 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                                             select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_8);

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                      select_half_operand<2, false, false>(packed_half_140),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                            select_half_operand<2, false, false>(packed_half_147));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_82 = result_word;
    }

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                      select_half_operand<2, false, false>(packed_half_141),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                       select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_accumulator_84 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_30 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                             select_half_operand<2, false, false>(packed_half_111));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_83 = result_word;
    }

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_30 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_74 = pack_half(matrix_fragment_73, matrix_fragment_74);

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_23 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_20), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_22 = result_word;
    }

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_85 = result_word;
    }

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                          select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 = permute_signed(matrix_accumulator_84, matrix_accumulator_84, 0x5432u);

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_accumulator_84 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                          select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_74 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_14 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5432u);

    matrix_accumulator_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84), 0x4100410u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_82 = half_to_float_bits((matrix_accumulator_84 & 65535u) * 65537u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_accumulator_84 = half_to_float_bits((matrix_accumulator_84 >> 16) * 65537u);

    matrix_fragment_23 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                        select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x4100410u);

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                       select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_fragment_75 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5432u);

    matrix_accumulator_82 = rsqrt_bits(matrix_accumulator_82);

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             packed_half_60, packed_half_61, 0u, 0u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_87 = rsqrt_bits(matrix_accumulator_84);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             packed_half_62, matrix_accumulator_63, 0u, 0u);

    matrix_accumulator_86 = half_to_float_bits((matrix_fragment_21 & 65535u) * 65537u);

    matrix_fragment_11 = pack_half(0u, matrix_fragment_11);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    matrix_accumulator_87 = pack_half(matrix_accumulator_87, matrix_accumulator_82);

    packed_half_107 = half_to_float_bits((matrix_fragment_21 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_92,
                             matrix_fragment_93, matrix_fragment_94, matrix_fragment_95,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_93 = result_word;
    }

    matrix_fragment_10 = permute_signed(matrix_fragment_23, matrix_fragment_23, 0x5432u);

    packed_half_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_86 = rsqrt_bits(matrix_accumulator_86);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_83 = result_word;
    }

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_164 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                    select_half_operand<2, false, false>(0u));

    packed_half_107 = rsqrt_bits(packed_half_107);

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    packed_half_157 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x4100410u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_97 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                                        select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_74 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x4100410u);

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_12 = threadIdx.x & 31u;

    packed_half_105 = permute_signed(matrix_fragment_92, matrix_fragment_92, 0x5432u);

    matrix_accumulator_85 = half_to_float_bits((matrix_accumulator_84 & 65535u) * 65537u);

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    packed_half_152 = half_to_float_bits((matrix_accumulator_84 >> 16) * 65537u);

    matrix_accumulator_86 = pack_half(packed_half_107, matrix_accumulator_86);

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_fragment_92 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                        select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_93 = half_to_float_bits((packed_half_153 & 65535u) * 65537u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_82 = half_to_float_bits((packed_half_153 >> 16) * 65537u);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_85 = rsqrt_bits(matrix_accumulator_85);

    packed_half_153 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_84 = half_to_float_bits((matrix_fragment_72 & 65535u) * 65537u);

    packed_half_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_accumulator_87));

    matrix_accumulator_83 = half_to_float_bits((matrix_fragment_72 >> 16) * 65537u);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x4100410u);

    packed_half_152 = rsqrt_bits(packed_half_152);

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(packed_half_164));

    matrix_accumulator_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(packed_half_157));

    packed_half_157 = half_to_float_bits((matrix_fragment_92 & 65535u) * 65537u);

    matrix_accumulator_84 = rsqrt_bits(matrix_accumulator_84);

    packed_half_164 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_92 = half_to_float_bits((matrix_fragment_92 >> 16) * 65537u);

    matrix_accumulator_83 = rsqrt_bits(matrix_accumulator_83);

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(packed_half_164));

    matrix_fragment_13 = rsqrt_bits(matrix_fragment_92);

    matrix_accumulator_85 = pack_half(packed_half_152, matrix_accumulator_85);

    split_address_words(address_word_158, address_word_159,
                        int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_158, address_word_159));

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x2c60ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_fragment_34 = pack_half(matrix_accumulator_83, matrix_accumulator_84);

    matrix_fragment_12 = rsqrt_bits(packed_half_157);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x2e60ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3060ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    packed_half_157 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                          select_half_operand<2, false, false>(matrix_fragment_34));

    thread_scratch[0] = matrix_fragment_13;

    matrix_fragment_35 = thread_scratch[0];

    matrix_fragment_93 = rsqrt_bits(matrix_fragment_93);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3260ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    matrix_accumulator_82 = rsqrt_bits(matrix_accumulator_82);

    packed_half_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(matrix_accumulator_86));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3460ull, matrix_accumulator_76,
        matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    packed_half_148 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_fragment_93 = pack_half(matrix_accumulator_82, matrix_fragment_93);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3660ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    packed_half_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    packed_half_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                            select_half_operand<2, false, false>(matrix_accumulator_85));

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3860ull, matrix_accumulator_84,
        matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87, memory_ranges,
        error_counts);

    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                      select_half_operand<2, false, false>(matrix_accumulator_97));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3a60ull, packed_half_104,
        packed_half_105, packed_half_106, packed_half_107, memory_ranges, error_counts);

    packed_half_164 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                          select_half_operand<2, false, false>(matrix_fragment_34));

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<2, false, false>(matrix_fragment_34));

    packed_half_165 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                          select_half_operand<2, false, false>(matrix_fragment_34));

    packed_half_123 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    packed_half_151 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<2, false, false>(matrix_fragment_93));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_half(matrix_fragment_35, matrix_fragment_12);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                       select_half_operand<2, false, false>(0u));

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    packed_half_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_28 = permute_signed(matrix_fragment_9, matrix_fragment_9, 0x5432u);

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x4100410u);

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_165),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = half_to_float_bits((matrix_fragment_9 & 65535u) * 65537u);

    matrix_fragment_25 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5432u);

    matrix_fragment_9 = half_to_float_bits((matrix_fragment_9 >> 16) * 65537u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x4100410u);

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_29);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_164),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                       select_half_operand<2, false, false>(0u));

    packed_half_149 = rsqrt_bits(matrix_fragment_9);

    matrix_accumulator_90 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_24 = half_to_float_bits((matrix_fragment_25 & 65535u) * 65537u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_9 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5432u);

    matrix_fragment_13 = half_to_float_bits((matrix_fragment_25 >> 16) * 65537u);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_123),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    multiply_accumulate_half(packed_half_122, packed_half_123, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_24 = rsqrt_bits(matrix_fragment_24);

    packed_half_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x4100410u);

    matrix_fragment_13 = rsqrt_bits(matrix_fragment_13);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = half_to_float_bits((matrix_fragment_27 & 65535u) * 65537u);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_20,
                             matrix_fragment_21);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x4100410u);

    matrix_fragment_21 = half_to_float_bits((matrix_fragment_27 >> 16) * 65537u);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                                       select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_20 = half_to_float_bits((matrix_fragment_25 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_22,
                             matrix_fragment_23);

    packed_half_0 = rsqrt_bits(packed_half_0);

    packed_half_148 = pack_half(matrix_fragment_13, matrix_fragment_24);

    matrix_fragment_22 = half_to_float_bits((matrix_fragment_25 >> 16) * 65537u);

    matrix_fragment_21 = rsqrt_bits(matrix_fragment_21);

    matrix_fragment_13 = rsqrt_bits(matrix_fragment_20);

    matrix_fragment_22 = rsqrt_bits(matrix_fragment_22);

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_21 = pack_half(matrix_fragment_21, packed_half_0);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_24 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                             select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                             select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_13 = pack_half(matrix_fragment_22, matrix_fragment_13);

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    packed_half_149 = pack_half(packed_half_149, matrix_fragment_10);

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                  select_half_operand<2, false, false>(0u));

    packed_half_141 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                          select_half_operand<2, false, false>(matrix_fragment_21));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_72,
                             matrix_fragment_73);

    packed_half_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                          select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_149));

    packed_half_0 = 0x29c0u;

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_11 = permute_signed(matrix_fragment_11, packed_half_0, 0x5410u);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_96,
                             matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_122),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0x3c203c20u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0x3c203c20u);

    packed_half_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_122), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    packed_half_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0x3e473e47u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0x3e473e47u);

    packed_half_122 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_122), 0x3e473e47u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_78,
                             matrix_accumulator_79);

    matrix_accumulator_136 = (packed_half_100 << 0x5u) + 0x7ff88000u;

    packed_half_0 = (packed_half_122 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_82,
                             matrix_accumulator_83);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_83 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 = (packed_half_110 << 0x5u) + 0x7ff88000u;

    packed_half_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_123),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_10 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                       select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                                        select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_84,
                             matrix_accumulator_85);

    packed_half_141 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_fragment_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_9),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0x3c203c20u);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_86,
                             matrix_accumulator_87);

    packed_half_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_123), 0x3c203c20u);

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0x3e473e47u);

    packed_half_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0x3e473e47u);

    matrix_accumulator_90 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3c203c20u);

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_24, matrix_fragment_25, packed_half_104,
                             packed_half_105);

    packed_half_123 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_123), 0x3e473e47u);

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x3c203c20u);

    matrix_accumulator_91 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3c203c20u);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_72,
                             matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                             matrix_fragment_22, matrix_fragment_23, packed_half_106,
                             packed_half_107);

    matrix_accumulator_90 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3e473e47u);

    matrix_fragment_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x3e473e47u);

    matrix_accumulator_91 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3e473e47u);

    matrix_accumulator_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_2),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_137 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_21 = (packed_half_101 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = (packed_half_111 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = (packed_half_123 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_140 = (matrix_accumulator_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                       select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_149 = (matrix_fragment_10 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_90 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    packed_half_148 = (matrix_accumulator_91 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x3c203c20u);

    matrix_accumulator_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_96 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3c203c20u);

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_accumulator_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_78 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3c203c20u);

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_accumulator_128 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3c203c20u);

    matrix_fragment_12 = threadIdx.x & 31u;

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_2 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2), 0x3e473e47u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                        select_half_operand<2, false, false>(packed_half_148));

    matrix_accumulator_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3e473e47u);

    matrix_accumulator_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_78 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3e473e47u);

    matrix_fragment_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    matrix_accumulator_128 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_128), 0x3e473e47u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_77 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3c203c20u);

    matrix_accumulator_79 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3c203c20u);

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    matrix_accumulator_129 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3c203c20u);

    matrix_fragment_10 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_3),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_145 = (matrix_accumulator_2 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_144 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_79 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3e473e47u);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                        select_half_operand<2, false, false>(packed_half_140));

    matrix_fragment_99 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_129 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_129), 0x3e473e47u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_156 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    packed_half_155 = (matrix_accumulator_78 << 0x5u) + 0x7ff88000u;

    packed_half_152 = (matrix_fragment_98 << 0x5u) + 0x7ff88000u;

    packed_half_151 = (matrix_accumulator_128 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_84 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x3c203c20u);

    matrix_accumulator_86 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3c203c20u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x3c203c20u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_97 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3c203c20u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_85 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85), 0x3c203c20u);

    matrix_accumulator_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_144));

    matrix_accumulator_87 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3c203c20u);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                                        select_half_operand<2, false, false>(packed_half_155));

    packed_half_157 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                        select_half_operand<2, false, false>(packed_half_151));

    packed_half_154 = (matrix_accumulator_79 << 0x5u) + 0x7ff88000u;

    packed_half_153 = (matrix_fragment_99 << 0x5u) + 0x7ff88000u;

    packed_half_150 = (matrix_accumulator_129 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_84 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x3e473e47u);

    matrix_accumulator_86 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3e473e47u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3c203c20u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3c203c20u);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x3e473e47u);

    matrix_accumulator_97 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3e473e47u);

    matrix_accumulator_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_accumulator_85 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85), 0x3e473e47u);

    matrix_accumulator_87 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3e473e47u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x3c203c20u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3c203c20u);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                                        select_half_operand<2, false, false>(packed_half_154));

    packed_half_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3e473e47u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                        select_half_operand<2, false, false>(packed_half_150));

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3e473e47u);

    packed_half_135 = (matrix_accumulator_84 << 0x5u) + 0x7ff88000u;

    packed_half_100 = (matrix_accumulator_86 << 0x5u) + 0x7ff88000u;

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x3e473e47u);

    packed_half_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3e473e47u);

    packed_half_147 = (matrix_fragment_10 << 0x5u) + 0x7ff88000u;

    packed_half_146 = (matrix_accumulator_97 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_20));

    packed_half_134 = (matrix_accumulator_85 << 0x5u) + 0x7ff88000u;

    packed_half_101 = (matrix_accumulator_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                        select_half_operand<2, false, false>(packed_half_100));

    packed_half_110 = (packed_half_104 << 0x5u) + 0x7ff88000u;

    packed_half_123 = (packed_half_106 << 0x5u) + 0x7ff88000u;

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                        select_half_operand<2, false, false>(packed_half_146));

    packed_half_111 = (packed_half_105 << 0x5u) + 0x7ff88000u;

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                        select_half_operand<2, false, false>(packed_half_101));

    packed_half_122 = (packed_half_107 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                        select_half_operand<2, false, false>(packed_half_123));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                        select_half_operand<2, false, false>(packed_half_122));

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 = matrix_fragment_12 * 0x4u;

    condition_0 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_60, packed_half_61, 0u, 0u);

    condition_1 = (matrix_fragment_12 & 0x2u) != 0;

    matrix_fragment_13 = matrix_fragment_13 & 0x1cu;

    matrix_fragment_10 =
        uint32_t(int64_t(int32_t(matrix_fragment_12)) * (1ll << 0x1du) >> 32) + matrix_fragment_13;

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_62, matrix_accumulator_63, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    matrix_fragment_4 = condition_0 ? matrix_fragment_9 : matrix_fragment_8;

    matrix_fragment_5 = condition_0 ? matrix_accumulator_3 : matrix_accumulator_2;

    matrix_fragment_8 = condition_0 ? matrix_fragment_8 : matrix_fragment_9;

    matrix_accumulator_3 = condition_0 ? matrix_accumulator_2 : matrix_accumulator_3;

    matrix_fragment_9 = 0x1u ^ matrix_fragment_10;

    matrix_fragment_13 = !condition_1 ? matrix_fragment_4 : matrix_fragment_5;

    matrix_fragment_5 = !condition_1 ? matrix_fragment_5 : matrix_fragment_4;

    matrix_accumulator_2 = !condition_1 ? matrix_fragment_8 : matrix_accumulator_3;

    matrix_fragment_4 = !condition_1 ? matrix_accumulator_3 : matrix_fragment_8;

    matrix_fragment_8 = 0x2u ^ matrix_fragment_10;

    matrix_fragment_7 = 0x3u ^ matrix_fragment_10;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_13), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    condition_2 = (matrix_fragment_12 & 0x8u) != 0;

    condition_3 = (matrix_fragment_12 & 0x10u) != 0;

    matrix_accumulator_3 = !condition_2 ? matrix_fragment_13 : matrix_accumulator_2;

    matrix_fragment_6 = !condition_2 ? matrix_accumulator_2 : matrix_fragment_13;

    matrix_fragment_20 = !condition_2 ? matrix_fragment_5 : matrix_fragment_4;

    matrix_fragment_13 = !condition_2 ? matrix_fragment_4 : matrix_fragment_5;

    matrix_accumulator_2 = !condition_3 ? matrix_accumulator_3 : matrix_fragment_20;

    matrix_fragment_5 = !condition_3 ? matrix_fragment_6 : matrix_fragment_13;

    matrix_accumulator_3 = !condition_3 ? matrix_fragment_20 : matrix_accumulator_3;

    matrix_fragment_6 = !condition_3 ? matrix_fragment_13 : matrix_fragment_6;

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_3),
                       select_half_operand<1, false, false>(matrix_accumulator_3));

    matrix_accumulator_3 = permute_signed(matrix_accumulator_3, matrix_accumulator_3, 0x5410u);

    matrix_accumulator_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3), 0x4100410u);

    matrix_accumulator_2 = half_to_float_bits((matrix_accumulator_3 & 65535u) * 65537u);

    matrix_accumulator_3 = half_to_float_bits((matrix_accumulator_3 >> 16) * 65537u);

    matrix_fragment_6 = matrix_fragment_12;

    matrix_accumulator_2 = reciprocal_bits(matrix_accumulator_2);

    matrix_fragment_5 = uint32_t(int32_t(matrix_fragment_6) >> 0x2u);

    matrix_accumulator_3 = reciprocal_bits(matrix_accumulator_3);

    matrix_fragment_4 = matrix_fragment_5 + 0x8u;

    matrix_fragment_13 = uint32_t(int32_t(matrix_fragment_4) >> 0x1fu);

    matrix_fragment_13 =
        uint32_t(combine_words(matrix_fragment_13, 0u) << 0x5u >> 32) + matrix_fragment_4;

    packed_half_165 = pack_half(matrix_accumulator_3, matrix_accumulator_2);

    matrix_accumulator_2 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_13 = matrix_fragment_13 & 0xffffffe0u;

    matrix_accumulator_77 = transpose_half_tile(matrix_accumulator_77);

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x5u >> 32) + matrix_fragment_5;

    matrix_fragment_72 = matrix_fragment_4 + (0u - matrix_fragment_13);

    matrix_accumulator_2 = matrix_accumulator_2 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_4 = matrix_fragment_5 + (0u - matrix_accumulator_2);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(matrix_fragment_72), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_accumulator_76 = transpose_half_tile(matrix_accumulator_76);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_20 = transpose_half_tile(matrix_accumulator_130);

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_accumulator_130 = transpose_half_tile(matrix_accumulator_79);

    matrix_fragment_13 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, packed_half_161) + 0x5070ull, memory_ranges, error_counts);

    matrix_accumulator_3 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_162, packed_half_163) + 0x5070ull, memory_ranges, error_counts);

    matrix_accumulator_2 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_142, matrix_accumulator_143) + 0x5070ull, memory_ranges,
        error_counts);

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_60, packed_half_61, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_62, matrix_accumulator_63, 0u, 0u);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    matrix_fragment_19 = transpose_half_tile(matrix_accumulator_78);

    packed_half_160 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4c70ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(0u));

    packed_half_161 = permute_signed(matrix_fragment_16, matrix_fragment_16, 0x5410u);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_83 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(packed_half_161));

    packed_half_0 = load_aligned_word<CheckBounds>(
        combine_words(address_word_138, address_word_139) + 0x5070ull, memory_ranges, error_counts);

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_accumulator_80));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4e70ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = transpose_half_tile(matrix_accumulator_133);

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_accumulator_77 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                      select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_131 = transpose_half_tile(matrix_accumulator_131);

    matrix_accumulator_76 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                      select_half_operand<2, false, false>(matrix_fragment_17));

    packed_half_109 = transpose_half_tile(packed_half_109);

    matrix_fragment_95 = transpose_half_tile(matrix_fragment_95);

    matrix_accumulator_97 = transpose_half_tile(matrix_accumulator_97);

    matrix_fragment_18 = transpose_half_tile(packed_half_108);

    matrix_fragment_17 = transpose_half_tile(matrix_accumulator_132);

    matrix_fragment_94 = transpose_half_tile(matrix_fragment_94);

    matrix_accumulator_96 = transpose_half_tile(matrix_accumulator_96);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    packed_half_108 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(0u));

    packed_half_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                      select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<2, false, false>(matrix_fragment_95));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_94 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_95 = uint32_t(int32_t(matrix_fragment_94) >> 0x1fu);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_95 =
        uint32_t(combine_words(matrix_fragment_95, 0u) << 0x5u >> 32) + matrix_fragment_94;

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_95 = matrix_fragment_95 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_78 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_accumulator_79 = uint32_t(int32_t(matrix_accumulator_78) >> 0x1fu);

    matrix_accumulator_79 =
        uint32_t(combine_words(matrix_accumulator_79, 0u) << 0x5u >> 32) + matrix_accumulator_78;

    matrix_accumulator_79 = matrix_accumulator_79 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    matrix_fragment_94 = matrix_fragment_94 + (0u - matrix_fragment_95);

    packed_half_164 = matrix_accumulator_78 + (0u - matrix_accumulator_79);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(matrix_fragment_94), "r"(0x1fu), "r"(__activemask()));
        packed_half_163 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(packed_half_164), "r"(0x1fu), "r"(__activemask()));
        packed_half_164 = result_word;
    }

    multiply_accumulate_half(address_word_138, address_word_139, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_60, packed_half_61, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             packed_half_62, matrix_accumulator_63, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    packed_half_107 = transpose_half_tile(packed_half_107);

    packed_half_162 = permute_signed(packed_half_164, packed_half_164, 0x5410u);

    matrix_accumulator_137 = transpose_half_tile(matrix_accumulator_137);

    packed_half_163 = permute_signed(packed_half_163, packed_half_163, 0x5410u);

    matrix_fragment_99 = transpose_half_tile(matrix_fragment_99);

    packed_half_106 = transpose_half_tile(packed_half_106);

    packed_half_155 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                          select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_136 = transpose_half_tile(matrix_accumulator_136);

    packed_half_151 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                          select_half_operand<2, false, false>(packed_half_162));

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                          select_half_operand<2, false, false>(packed_half_163));

    matrix_fragment_98 = transpose_half_tile(matrix_fragment_98);

    packed_half_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                          select_half_operand<2, false, false>(packed_half_163));

    packed_half_105 = transpose_half_tile(packed_half_105);

    multiply_accumulate_half(matrix_fragment_94, matrix_fragment_95, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_accumulator_143 = transpose_half_tile(matrix_accumulator_143);

    address_word_139 = transpose_half_tile(address_word_139);

    matrix_fragment_52 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                                             select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_129 = transpose_half_tile(matrix_accumulator_129);

    matrix_fragment_53 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                                             select_half_operand<2, false, false>(packed_half_163));

    matrix_accumulator_79 = transpose_half_tile(matrix_accumulator_79);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    packed_half_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                    select_half_operand<2, false, false>(0u));

    packed_half_104 = transpose_half_tile(packed_half_104);

    packed_half_151 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                    select_half_operand<2, false, false>(0u));

    packed_half_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_142 = transpose_half_tile(matrix_accumulator_142);

    matrix_fragment_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                             select_half_operand<2, false, false>(packed_half_162));

    packed_half_150 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                             select_half_operand<2, false, false>(packed_half_163));

    address_word_138 = transpose_half_tile(address_word_138);

    matrix_accumulator_128 = transpose_half_tile(matrix_accumulator_128);

    matrix_accumulator_78 = transpose_half_tile(matrix_accumulator_78);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_150));

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_140 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                          select_half_operand<2, false, false>(packed_half_161));

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(packed_half_161));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                    select_half_operand<2, false, false>(0u));

    packed_half_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_148 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(0u));

    packed_half_141 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_141),
                                          select_half_operand<2, false, false>(packed_half_161));

    packed_half_146 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                    select_half_operand<2, false, false>(0u));

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_137 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(packed_half_107));

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_52, matrix_fragment_53,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_18,
                             matrix_fragment_19, 0u, 0u);

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                    select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_143),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    address_word_139 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_139),
                                     select_half_operand<2, false, false>(0u));

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                       select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_141),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_138),
                                    select_half_operand<2, false, false>(address_word_139));

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_106, packed_half_107, packed_half_108, packed_half_109);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                          select_half_operand<2, false, false>(packed_half_162));

    packed_half_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<2, false, false>(packed_half_162));

    packed_half_108 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                    select_half_operand<2, false, false>(matrix_accumulator_129));

    packed_half_109 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                    select_half_operand<2, false, false>(matrix_accumulator_79));

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                                          select_half_operand<2, false, false>(packed_half_163));

    packed_half_122 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                          select_half_operand<2, false, false>(packed_half_163));

    packed_half_123 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_123),
                                    select_half_operand<2, false, false>(0u));

    packed_half_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<2, false, false>(packed_half_162));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_130,
                             matrix_accumulator_131);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<2, false, false>(packed_half_162));

    matrix_accumulator_78 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                          select_half_operand<2, false, false>(0u));

    packed_half_122 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_104, packed_half_105, matrix_accumulator_132,
                             matrix_accumulator_133);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_108, packed_half_109, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_53 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                             select_half_operand<2, false, false>(packed_half_163));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                             select_half_operand<2, false, false>(packed_half_163));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(packed_half_123));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_122));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_98, matrix_fragment_99, packed_half_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_108, packed_half_109, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_fragment_52, matrix_fragment_53,
                             matrix_fragment_54, matrix_fragment_55, packed_half_104,
                             packed_half_105, packed_half_60, packed_half_61);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_106, packed_half_107, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_122, packed_half_123, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    packed_half_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_60),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_61),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    packed_half_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(packed_half_62));

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                                         select_half_operand<2, false, false>(matrix_fragment_92));

    matrix_accumulator_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(packed_half_60));

    matrix_accumulator_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(packed_half_62));

    packed_half_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(matrix_fragment_93));

    packed_half_60 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                         select_half_operand<2, false, false>(packed_half_110));

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                         select_half_operand<2, false, false>(packed_half_122));

    packed_half_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                      select_half_operand<2, false, false>(matrix_accumulator_85),
                                      select_half_operand<2, false, false>(packed_half_61));

    matrix_accumulator_63 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                      select_half_operand<2, false, false>(matrix_accumulator_87),
                                      select_half_operand<2, false, false>(matrix_accumulator_63));

    packed_half_60 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                                      select_half_operand<2, false, false>(matrix_fragment_94),
                                      select_half_operand<2, false, false>(packed_half_60));

    packed_half_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(packed_half_62));

    packed_half_61 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                                    select_half_operand<2, false, false>(packed_half_61));

    packed_half_60 = add_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                                    select_half_operand<2, false, false>(packed_half_60));

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                         select_half_operand<2, false, false>(packed_half_123));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                            select_half_operand<2, false, false>(packed_half_111));

    packed_half_62 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(packed_half_62));

    matrix_accumulator_130 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                                      select_half_operand<2, false, false>(matrix_fragment_95),
                                      select_half_operand<2, false, false>(matrix_accumulator_63));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                          select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_124),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_125),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                       select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                       select_half_operand<2, false, false>(matrix_accumulator_130));

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_72, matrix_fragment_73, packed_half_112,
                             packed_half_113);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_60), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_131 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_63 = result_word;
    }

    uniform_index_word_5 = blockIdx.y;

    multiply_accumulate_half(packed_half_114, matrix_accumulator_115, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_74, matrix_fragment_75, packed_half_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_118,
                             matrix_accumulator_119);

    multiply_accumulate_half(matrix_fragment_124, matrix_fragment_125, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_124,
                             matrix_fragment_125);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_61), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_52 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_130), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    uniform_index_word_20 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);

    matrix_accumulator_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    uniform_index_word_20 =
        uint32_t(combine_words(uniform_index_word_20, 0u) << 0x2u >> 32) + uniform_index_word_10;

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    packed_half_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_103),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_accumulator_132 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_0));

    packed_half_135 = add_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                                     select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_131 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                       select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                       select_half_operand<2, false, false>(matrix_fragment_53));

    uniform_index_word_20 = uint32_t(int32_t(uniform_index_word_20) >> 0x2u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_132), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_137 = result_word;
    }

    uniform_block_index_18 = (uniform_index_word_5 << 0x3u) + uniform_block_index_18;

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_72, matrix_fragment_73, matrix_accumulator_116,
                             matrix_accumulator_117);

    uniform_condition_1 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_20);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_135), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_136 = result_word;
    }

    uniform_index_word_19 = uint32_t(int32_t(uniform_index_word_11) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_131), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_134 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_130), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_133 = result_word;
    }

    uniform_condition_1 =
        int32_t(uniform_block_index_18) > int32_t(0u - 0x4u) && !uniform_condition_1;

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_74, matrix_fragment_75, matrix_accumulator_120,
                             matrix_accumulator_121);

    uniform_index_word_19 =
        uint32_t(combine_words(uniform_index_word_19, 0u) << 0x2u >> 32) + uniform_index_word_11;

    uniform_condition_2 = uniform_condition_1 & uniform_condition_6;

    uniform_index_word_19 = uint32_t(int32_t(uniform_index_word_19) >> 0x2u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_102,
                             packed_half_103);

    uniform_index_word_5 = uniform_index_word_12 * uniform_index_word_19 + uniform_index_word_13;

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_82, matrix_accumulator_83, matrix_accumulator_126,
                             matrix_accumulator_127);

    uniform_index_word_5 = uniform_index_word_5 << (0x7u & 31u);

    split_address_words(uniform_address_word_14, uniform_parameter_word_15,
                        int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_14, uniform_parameter_word_15));
    if (uniform_condition_2) {
        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_124),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_56, matrix_fragment_57,
                            combine_words(uniform_address_word_14, uniform_parameter_word_15));

        matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(matrix_fragment_53));

        split_address_words(matrix_fragment_56, matrix_fragment_57,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_56, matrix_fragment_57));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                          select_half_operand<2, false, false>(matrix_fragment_52));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                          select_half_operand<2, false, false>(matrix_fragment_59));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_56, matrix_fragment_57), matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);
    }
    uniform_condition_1 = uniform_condition_1 & uniform_condition_0;

    split_address_words(address_word_138, address_word_139,
                        combine_words(uniform_address_word_14, uniform_parameter_word_15));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                    select_half_operand<2, false, false>(0u));

    uniform_condition_2 = int32_t(uniform_index_word_22) != int32_t(0u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                                       select_half_operand<2, false, false>(0u));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(0u));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                       select_half_operand<2, false, false>(matrix_accumulator_137));

    condition_4 = uniform_condition_1;

    packed_half_135 = add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                     select_half_operand<2, false, false>(matrix_accumulator_136));

    uniform_parameter_word_15 = parameters[parameter_scalar_control_9];

    matrix_accumulator_131 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                       select_half_operand<2, false, false>(packed_half_134));

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                       select_half_operand<2, false, false>(matrix_accumulator_133));

    uniform_index_word_4 = parameters[parameter_output_low_2];
    uniform_index_word_5 = parameters[parameter_output_high_3];

    if (condition_4) {
        packed_half_103 = permute_signed(packed_half_103, matrix_fragment_54, 0x5410u);

        split_address_words(address_word_138, address_word_139,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_138, address_word_139));

        matrix_accumulator_102 =
            permute_signed(matrix_accumulator_102, matrix_fragment_53, 0x5410u);

        packed_half_101 = permute_signed(packed_half_101, matrix_fragment_52, 0x5410u);

        packed_half_100 = permute_signed(matrix_fragment_55, packed_half_100, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(address_word_138, address_word_139) + 0x200ull, packed_half_100,
            packed_half_101, matrix_accumulator_102, packed_half_103, memory_ranges, error_counts);
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3c60ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x3e60ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4060ull, packed_half_60,
        packed_half_61, packed_half_62, matrix_accumulator_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4260ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4460ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4660ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4860ull, matrix_accumulator_76,
        matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4a60ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    packed_half_101 = permute_signed(matrix_accumulator_132, matrix_accumulator_132, 0x5432u);

    matrix_accumulator_102 = permute_signed(packed_half_135, packed_half_135, 0x5432u);

    matrix_accumulator_132 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                       select_half_operand<2, false, false>(packed_half_101));

    packed_half_135 = add_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                     select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x4100410u);

    packed_half_135 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_135), 0x4100410u);

    packed_half_100 = half_to_float_bits((matrix_accumulator_132 & 65535u) * 65537u);

    matrix_accumulator_102 = half_to_float_bits((packed_half_135 & 65535u) * 65537u);

    packed_half_135 = half_to_float_bits((packed_half_135 >> 16) * 65537u);

    matrix_accumulator_132 = half_to_float_bits((matrix_accumulator_132 >> 16) * 65537u);

    packed_half_100 = rsqrt_bits(packed_half_100);

    packed_half_101 = rsqrt_bits(matrix_accumulator_132);

    matrix_accumulator_102 = rsqrt_bits(matrix_accumulator_102);

    packed_half_135 = rsqrt_bits(packed_half_135);

    packed_half_101 = pack_half(packed_half_101, packed_half_100);

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    packed_half_103 = pack_half(packed_half_135, matrix_accumulator_102);

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(packed_half_103));

    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(packed_half_103));

    matrix_accumulator_90 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(packed_half_103));

    matrix_accumulator_84 = permute_signed(matrix_accumulator_84, packed_half_101, 0x5410u);

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<2, false, false>(packed_half_103));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_92),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_86 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_87 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_92 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_85 = permute_signed(matrix_accumulator_85, matrix_accumulator_86, 0x5410u);

    matrix_accumulator_86 = permute_signed(matrix_accumulator_90, packed_half_103, 0x5410u);

    matrix_accumulator_87 = permute_signed(matrix_accumulator_87, matrix_fragment_92, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_52 = permute_signed(matrix_accumulator_131, matrix_accumulator_131, 0x5432u);

    matrix_fragment_53 = permute_signed(matrix_accumulator_130, matrix_accumulator_130, 0x5432u);

    matrix_accumulator_131 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_accumulator_130 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                       select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_fragment_92, matrix_fragment_93, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_accumulator_131 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_131), 0x4100410u);

    matrix_accumulator_130 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_130), 0x4100410u);

    matrix_fragment_52 = half_to_float_bits((matrix_accumulator_131 & 65535u) * 65537u);

    matrix_accumulator_131 = half_to_float_bits((matrix_accumulator_131 >> 16) * 65537u);

    matrix_fragment_53 = half_to_float_bits((matrix_accumulator_130 & 65535u) * 65537u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_130 = half_to_float_bits((matrix_accumulator_130 >> 16) * 65537u);

    matrix_fragment_52 = rsqrt_bits(matrix_fragment_52);

    matrix_accumulator_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_accumulator_90 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3c203c20u);

    matrix_accumulator_91 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3c203c20u);

    matrix_accumulator_131 = rsqrt_bits(matrix_accumulator_131);

    matrix_accumulator_90 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3e473e47u);

    matrix_accumulator_91 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x3e473e47u);

    matrix_fragment_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_92),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_53 = rsqrt_bits(matrix_fragment_53);

    matrix_fragment_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_93),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3c203c20u);

    matrix_fragment_93 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3c203c20u);

    matrix_accumulator_130 = rsqrt_bits(matrix_accumulator_130);

    matrix_fragment_92 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_92), 0x3e473e47u);

    matrix_fragment_55 = pack_half(matrix_accumulator_131, matrix_fragment_52);

    matrix_fragment_93 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_93), 0x3e473e47u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(packed_half_60, packed_half_61, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_28, matrix_fragment_29, packed_half_60,
                             packed_half_61);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    packed_half_122 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                          select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_94),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_52 = pack_half(matrix_accumulator_130, matrix_fragment_53);

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                    select_half_operand<2, false, false>(0u));

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<2, false, false>(matrix_fragment_52));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_26, matrix_fragment_27, packed_half_62,
                             matrix_accumulator_63);

    packed_half_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_94),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_95),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_53, 0x5410u);

    packed_half_122 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_95 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_95),
                                       select_half_operand<2, false, false>(0u));

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                       select_half_operand<2, false, false>(0u));

    packed_half_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<0, false, false>(matrix_fragment_11));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                                       select_half_operand<2, false, false>(0u));

    packed_half_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_60),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                    select_half_operand<2, false, false>(0u));

    packed_half_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_61),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_53 = permute_signed(matrix_fragment_95, matrix_fragment_54, 0x5410u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_54 = permute_signed(packed_half_110, matrix_fragment_55, 0x5410u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    packed_half_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x3c203c20u);

    packed_half_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x3c203c20u);

    packed_half_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_60), 0x3e473e47u);

    packed_half_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_61), 0x3e473e47u);

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_accumulator_84,
                             matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_66,
                             matrix_accumulator_67);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x3c203c20u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x3e473e47u);

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    matrix_accumulator_84 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_55 = permute_signed(packed_half_111, matrix_accumulator_84, 0x5410u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_36 = (matrix_accumulator_90 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_37 = (matrix_accumulator_91 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_accumulator_66 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_33 = (matrix_fragment_33 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_73),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_32 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_72),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3c203c20u);

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3c203c20u);

    matrix_fragment_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x3e473e47u);

    matrix_fragment_72 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x3e473e47u);

    matrix_fragment_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_75),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_34 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_74),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_72 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_78,
                             matrix_accumulator_79);

    matrix_fragment_75 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3c203c20u);

    matrix_fragment_74 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3c203c20u);

    matrix_fragment_75 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75), 0x3e473e47u);

    matrix_fragment_74 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74), 0x3e473e47u);

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_35 = (matrix_fragment_74 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_fragment_28 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_accumulator_77 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3c203c20u);

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_accumulator_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3e473e47u);

    matrix_accumulator_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3e473e47u);

    matrix_fragment_57 = (matrix_fragment_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_58 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_23 = (packed_half_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_22 = (packed_half_61 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_54 = (packed_half_62 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_53 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    packed_half_60 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    packed_half_61 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    packed_half_62 = (matrix_fragment_73 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_63 = (matrix_fragment_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = add_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                        select_half_operand<2, false, false>(packed_half_61));

    matrix_accumulator_85 = (matrix_fragment_24 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_accumulator_78 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3c203c20u);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_accumulator_79 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3c203c20u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<1, false, false>(matrix_fragment_11), 0x3d343d34u);

    matrix_fragment_68 = (matrix_fragment_25 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_accumulator_78 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3e473e47u);

    matrix_fragment_25 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                       select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_accumulator_79 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3e473e47u);

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_56 = (matrix_fragment_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_59 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                        select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                        select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_69 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                        select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_67 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_52 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_28 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_55 = (matrix_accumulator_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4c70ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_64 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = (matrix_accumulator_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_74 = (matrix_accumulator_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_69 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_72 = (matrix_fragment_72 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_66 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_73 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_75),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                                        select_half_operand<2, false, false>(matrix_fragment_74));

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                        select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                       select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_28 = condition_0 ? matrix_fragment_30 : matrix_fragment_11;

    matrix_fragment_29 = condition_0 ? matrix_accumulator_76 : matrix_fragment_31;

    matrix_fragment_11 = condition_0 ? matrix_fragment_11 : matrix_fragment_30;

    matrix_accumulator_76 = condition_0 ? matrix_fragment_31 : matrix_accumulator_76;

    matrix_fragment_75 = !condition_1 ? matrix_fragment_28 : matrix_fragment_29;

    matrix_accumulator_77 = !condition_1 ? matrix_fragment_29 : matrix_fragment_28;

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x4e70ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_accumulator_78 = !condition_1 ? matrix_fragment_11 : matrix_accumulator_76;

    matrix_accumulator_76 = !condition_1 ? matrix_accumulator_76 : matrix_fragment_11;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_75), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_78), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_77), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_77 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_76), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_76 = result_word;
    }

    matrix_fragment_11 = !condition_2 ? matrix_fragment_75 : matrix_accumulator_78;

    matrix_accumulator_79 = !condition_2 ? matrix_accumulator_78 : matrix_fragment_75;

    matrix_accumulator_80 = !condition_2 ? matrix_accumulator_77 : matrix_accumulator_76;

    matrix_accumulator_82 = !condition_2 ? matrix_accumulator_76 : matrix_accumulator_77;

    matrix_accumulator_81 = !condition_3 ? matrix_fragment_11 : matrix_accumulator_80;

    matrix_accumulator_84 = !condition_3 ? matrix_accumulator_79 : matrix_accumulator_82;

    matrix_fragment_11 = !condition_3 ? matrix_accumulator_80 : matrix_fragment_11;

    matrix_accumulator_79 = !condition_3 ? matrix_accumulator_82 : matrix_accumulator_79;

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_81),
                       select_half_operand<1, false, false>(matrix_accumulator_81));

    matrix_accumulator_81 = permute_signed(matrix_accumulator_81, matrix_accumulator_81, 0x5410u);

    matrix_accumulator_81 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81), 0x4100410u);

    matrix_fragment_7 = half_to_float_bits((matrix_accumulator_81 & 65535u) * 65537u);

    matrix_accumulator_81 = half_to_float_bits((matrix_accumulator_81 >> 16) * 65537u);

    matrix_fragment_7 = reciprocal_bits(matrix_fragment_7);

    matrix_fragment_10 = matrix_fragment_5 + 0x8u;

    matrix_fragment_8 = reciprocal_bits(matrix_accumulator_81);

    matrix_fragment_9 = uint32_t(int32_t(matrix_fragment_10) >> 0x1fu);

    matrix_fragment_9 =
        uint32_t(combine_words(matrix_fragment_9, 0u) << 0x5u >> 32) + matrix_fragment_10;

    matrix_fragment_75 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_11 = matrix_fragment_9 & 0xffffffe0u;

    matrix_accumulator_76 = uint32_t(int32_t(matrix_fragment_75) >> 0x1fu);

    matrix_fragment_10 = matrix_fragment_10 + (0u - matrix_fragment_11);

    matrix_fragment_9 = pack_half(matrix_fragment_8, matrix_fragment_7);

    matrix_fragment_8 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_11 = uint32_t(int32_t(matrix_fragment_8) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_accumulator_76 =
        uint32_t(combine_words(matrix_accumulator_76, 0u) << 0x5u >> 32) + matrix_fragment_75;

    matrix_fragment_11 =
        uint32_t(combine_words(matrix_fragment_11, 0u) << 0x5u >> 32) + matrix_fragment_8;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_7 = result_word;
    }

    matrix_accumulator_78 = matrix_accumulator_76 & 0xffffffe0u;

    matrix_fragment_11 = matrix_fragment_11 & 0xffffffe0u;

    matrix_accumulator_78 = matrix_fragment_75 + (0u - matrix_accumulator_78);

    matrix_accumulator_76 = matrix_fragment_8 + (0u - matrix_fragment_11);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_78), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_76), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_75 = result_word;
    }

    matrix_fragment_4 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5410u);

    matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_7, 0x5410u);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_75 = permute_signed(matrix_fragment_75, matrix_fragment_75, 0x5410u);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_78 = permute_signed(matrix_accumulator_78, matrix_accumulator_78, 0x5410u);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    packed_half_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_61),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_74),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_fragment_75));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_104, packed_half_105, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_56,
                             matrix_fragment_57);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_106, packed_half_107, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_108, packed_half_109, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_20,
                             matrix_fragment_21);

    uniform_address_word_14 = blockIdx.y;

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, packed_half_108,
                             packed_half_109, matrix_fragment_14, matrix_fragment_15);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_104, packed_half_105, matrix_fragment_18,
                             matrix_fragment_19);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, matrix_fragment_11, packed_half_106,
                             packed_half_107, matrix_fragment_16, matrix_fragment_17);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_50),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_51),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    packed_half_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_accumulator_3));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_0));

    uniform_index_word_7 = uniform_index_word_12 + 0x1u;

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_36,
                             matrix_fragment_37);

    uniform_index_word_6 = (uniform_address_word_14 << 0x3u) + uniform_parameter_word_15;

    uniform_condition_1 = int32_t(uniform_index_word_7) >= int32_t(uniform_index_word_20);

    uniform_condition_1 =
        int32_t(uniform_index_word_6) > int32_t(0u - 0x8u) && !uniform_condition_1;

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_42,
                             matrix_fragment_43);

    uniform_condition_2 = uniform_condition_1 & uniform_condition_2;

    uniform_index_word_12 = uniform_index_word_12 * uniform_index_word_19 + uniform_index_word_19;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_14,
                             matrix_fragment_15);

    uniform_index_word_13 = uniform_index_word_13 + uniform_index_word_12;

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, packed_half_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_38,
                             matrix_fragment_39);

    uniform_index_word_13 = uniform_index_word_13 << (0x7u & 31u);

    split_address_words(uniform_index_word_4, uniform_index_word_5,
                        int64_t(int32_t(uniform_index_word_13)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_index_word_4, uniform_index_word_5));
    if (uniform_condition_2) {
        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            combine_words(uniform_index_word_4, uniform_index_word_5));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                          select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                           select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                           select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_1 & uniform_condition_0;
    if (uniform_condition_0) {
        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            combine_words(uniform_index_word_4, uniform_index_word_5));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                           select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                           select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(matrix_fragment_7));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x200ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x54b0ull, matrix_fragment_16,
        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x50b0ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x52b0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_158, address_word_159) + 0x56b0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    condition_0 = (matrix_fragment_12 & 0x4u) != 0;

    packed_half_0 = uint32_t(uint32_t(matrix_fragment_12) >> 0x1u);

    matrix_accumulator_3 = matrix_fragment_12 & 0x13u;

    uniform_block_index_18 = blockIdx.x;

    matrix_fragment_4 = !condition_0 ? packed_half_114 : matrix_accumulator_120;

    matrix_fragment_13 = !condition_0 ? matrix_accumulator_115 : matrix_accumulator_121;

    uniform_index_word_8 = parameters[parameter_optional_control_8];

    matrix_fragment_38 = !condition_3 ? matrix_fragment_4 : matrix_fragment_13;

    matrix_fragment_39 = !condition_3 ? matrix_fragment_13 : matrix_fragment_4;

    matrix_fragment_4 = !condition_0 ? matrix_accumulator_118 : matrix_accumulator_128;

    matrix_fragment_13 = !condition_0 ? matrix_accumulator_119 : matrix_accumulator_129;

    matrix_fragment_41 = !condition_0 ? matrix_fragment_124 : matrix_accumulator_126;

    matrix_fragment_44 = !condition_0 ? matrix_fragment_125 : matrix_accumulator_127;

    packed_half_50 = !condition_3 ? matrix_fragment_4 : matrix_fragment_13;

    matrix_fragment_45 = !condition_3 ? matrix_fragment_13 : matrix_fragment_4;

    matrix_fragment_4 = !condition_0 ? matrix_fragment_14 : matrix_fragment_32;

    matrix_fragment_13 = !condition_0 ? matrix_fragment_15 : matrix_fragment_33;

    matrix_fragment_14 = !condition_0 ? matrix_fragment_32 : matrix_fragment_14;

    matrix_fragment_15 = !condition_0 ? matrix_fragment_33 : matrix_fragment_15;

    matrix_accumulator_3 = matrix_accumulator_3 | 0x4u & packed_half_0;

    packed_half_0 = matrix_fragment_12 * 0x2u;

    matrix_fragment_54 = !condition_3 ? matrix_fragment_41 : matrix_fragment_44;

    packed_half_51 = !condition_3 ? matrix_fragment_44 : matrix_fragment_41;

    matrix_fragment_41 = !condition_0 ? matrix_fragment_34 : packed_half_46;

    packed_half_46 = !condition_0 ? packed_half_46 : matrix_fragment_34;

    matrix_fragment_34 = !condition_3 ? matrix_fragment_14 : matrix_fragment_15;

    packed_half_114 = !condition_0 ? matrix_accumulator_120 : packed_half_114;

    matrix_accumulator_115 = !condition_0 ? matrix_accumulator_121 : matrix_accumulator_115;

    matrix_fragment_32 = !condition_3 ? matrix_fragment_4 : matrix_fragment_13;

    matrix_fragment_33 = !condition_3 ? matrix_fragment_13 : matrix_fragment_4;

    matrix_fragment_14 = !condition_3 ? matrix_fragment_15 : matrix_fragment_14;

    matrix_fragment_7 = matrix_accumulator_3 | 0x8u & packed_half_0;

    matrix_fragment_4 = !condition_0 ? matrix_fragment_36 : matrix_fragment_28;

    matrix_fragment_13 = !condition_0 ? matrix_fragment_37 : matrix_fragment_29;

    matrix_fragment_15 = !condition_0 ? matrix_fragment_42 : matrix_fragment_30;

    matrix_fragment_58 = !condition_0 ? matrix_fragment_43 : matrix_fragment_31;

    packed_half_0 = !condition_0 ? packed_half_112 : matrix_accumulator_116;

    matrix_accumulator_3 = !condition_0 ? packed_half_113 : matrix_accumulator_117;

    matrix_fragment_44 = !condition_0 ? matrix_fragment_35 : matrix_fragment_47;

    packed_half_112 = !condition_0 ? matrix_accumulator_116 : packed_half_112;

    packed_half_113 = !condition_0 ? matrix_accumulator_117 : packed_half_113;

    matrix_fragment_40 = !condition_3 ? packed_half_114 : matrix_accumulator_115;

    matrix_fragment_35 = !condition_0 ? matrix_fragment_47 : matrix_fragment_35;

    matrix_fragment_42 = !condition_0 ? matrix_fragment_30 : matrix_fragment_42;

    packed_half_114 = !condition_3 ? matrix_accumulator_115 : packed_half_114;

    matrix_fragment_30 = !condition_3 ? matrix_fragment_4 : matrix_fragment_13;

    matrix_fragment_47 = !condition_3 ? matrix_fragment_13 : matrix_fragment_4;

    packed_half_60 = !condition_3 ? matrix_fragment_15 : matrix_fragment_58;

    matrix_fragment_55 = !condition_3 ? matrix_fragment_58 : matrix_fragment_15;

    matrix_accumulator_2 = !condition_3 ? packed_half_0 : matrix_accumulator_3;

    matrix_fragment_13 = 0x4u ^ matrix_fragment_7;

    matrix_fragment_4 = 0x10u ^ matrix_fragment_7;

    matrix_fragment_15 = 0x14u ^ matrix_fragment_7;

    matrix_accumulator_3 = !condition_3 ? matrix_accumulator_3 : packed_half_0;

    matrix_fragment_48 = !condition_3 ? matrix_fragment_41 : matrix_fragment_44;

    packed_half_0 = !condition_3 ? packed_half_112 : packed_half_113;

    matrix_fragment_41 = !condition_3 ? matrix_fragment_44 : matrix_fragment_41;

    matrix_fragment_28 = !condition_0 ? matrix_fragment_28 : matrix_fragment_36;

    matrix_fragment_29 = !condition_0 ? matrix_fragment_29 : matrix_fragment_37;

    packed_half_112 = !condition_3 ? packed_half_113 : packed_half_112;

    matrix_fragment_44 = !condition_3 ? packed_half_46 : matrix_fragment_35;

    packed_half_46 = !condition_3 ? matrix_fragment_35 : packed_half_46;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_38), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    matrix_fragment_36 = !condition_3 ? matrix_fragment_28 : matrix_fragment_29;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    matrix_fragment_28 = !condition_3 ? matrix_fragment_29 : matrix_fragment_28;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_39), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_114), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        packed_half_114 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_2), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_2 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_29 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_3), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_3 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_112), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        packed_half_112 = result_word;
    }

    matrix_fragment_124 = !condition_0 ? matrix_accumulator_126 : matrix_fragment_124;

    matrix_fragment_125 = !condition_0 ? matrix_accumulator_127 : matrix_fragment_125;

    matrix_fragment_31 = !condition_0 ? matrix_fragment_31 : matrix_fragment_43;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_accumulator_118 = !condition_0 ? matrix_accumulator_128 : matrix_accumulator_118;

    matrix_accumulator_119 = !condition_0 ? matrix_accumulator_129 : matrix_accumulator_119;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_28), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_28 = result_word;
    }

    matrix_fragment_56 = !condition_3 ? matrix_fragment_124 : matrix_fragment_125;

    matrix_fragment_58 = !condition_3 ? matrix_fragment_42 : matrix_fragment_31;

    matrix_fragment_124 = !condition_3 ? matrix_fragment_125 : matrix_fragment_124;

    matrix_fragment_42 = !condition_3 ? matrix_fragment_31 : matrix_fragment_42;

    matrix_fragment_52 = !condition_3 ? matrix_accumulator_118 : matrix_accumulator_119;

    matrix_accumulator_118 = !condition_3 ? matrix_accumulator_119 : matrix_accumulator_118;

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                        select_half_operand<2, false, false>(packed_half_114));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_48), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_fragment_29));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_54), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_54 = result_word;
    }

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_60), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        packed_half_60 = result_word;
    }

    matrix_accumulator_3 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                       select_half_operand<2, false, false>(packed_half_112));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_44), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_37 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_56), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_58), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_41), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_46), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        packed_half_46 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_51), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        packed_half_51 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_124), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_124 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_55), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_55 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_42), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_accumulator_2 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                       select_half_operand<2, false, false>(matrix_accumulator_3));

    uniform_index_word_7 = (uniform_block_index_18 << 0x3u) + uniform_index_word_8;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_32), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_32 = result_word;
    }

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_50), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        packed_half_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_30), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_30 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_34), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_52), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_36), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_33), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_33 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_45), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_118), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_118 = result_word;
    }

    matrix_accumulator_2 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_2), 0x34003400u);

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    uniform_index_word_7 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x1u >> 32) + uniform_index_word_7;

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 =
        uint32_t(combine_words(matrix_fragment_5, 0u) << 0x2u >> 32) + matrix_fragment_5;

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_6) >> 0x1u);

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_7) >> 0x1u);

    uniform_index_word_4 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x1u >> 32) + uniform_index_word_10;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_2),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x1u >> 32) + uniform_index_word_11;

    matrix_accumulator_2 = matrix_fragment_12;

    packed_half_0 = packed_half_0 & 0xfffffffcu;

    uniform_index_word_4 = uint32_t(int32_t(uniform_index_word_4) >> 0x1u);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_5) >> 0x1u);

    matrix_accumulator_2 = uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1cu) >> 32) +
                           uniform_index_word_6;

    packed_half_0 = matrix_fragment_5 + uniform_index_word_7 + (0u - packed_half_0);

    matrix_fragment_37 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(packed_half_46));

    condition_0 = int32_t(matrix_accumulator_2) >= int32_t(uniform_index_word_4);

    matrix_fragment_53 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    condition_1 = int32_t(packed_half_0) >= int32_t(uniform_index_word_5);

    packed_half_51 = add_half_pairs(select_half_operand<2, false, false>(packed_half_51),
                                    select_half_operand<2, false, false>(matrix_fragment_124));

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(packed_half_60),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_4 = matrix_accumulator_2 + 0x2u;

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(packed_half_50),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_45 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                       select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_30 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_37 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    packed_half_51 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                    select_half_operand<2, false, false>(packed_half_51));

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    condition_0 = int32_t(matrix_accumulator_2) < int32_t(0u) || condition_0;

    condition_1 = int32_t(packed_half_0) < int32_t(0u) || condition_1;

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    condition_2 = int32_t(matrix_fragment_4) >= int32_t(uniform_index_word_4);

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x34003400u);

    packed_half_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_51), 0x34003400u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x34003400u);

    condition_0 = condition_0 | condition_1;

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x34003400u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x34003400u);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x34003400u);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    condition_2 = int32_t(matrix_fragment_4) < int32_t(0u) || condition_2;

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    condition_2 = condition_2 | condition_1;

    uniform_index_word_5 = uniform_index_word_5 << (0x2u & 31u);

    if (!condition_0) {
        matrix_fragment_5 = matrix_accumulator_2 + uniform_index_word_4;

        matrix_fragment_14 = packed_half_0 * 0x4u;

        matrix_accumulator_3 = packed_half_0 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    if (!condition_0) {
        matrix_fragment_7 = matrix_fragment_5 * uniform_index_word_5 + matrix_fragment_14;

        matrix_accumulator_3 = matrix_accumulator_2 * uniform_index_word_5 + matrix_accumulator_3;
    }

    matrix_fragment_17 = uniform_index_word_4;

    if (!condition_2) {
        matrix_fragment_5 = packed_half_0 * 0x4u;

        matrix_fragment_13 = packed_half_0 * 0x4u;
    }

    if (!condition_0) {
        matrix_fragment_14 = matrix_fragment_17 * 0x2u + matrix_accumulator_2;

        matrix_fragment_16 = matrix_fragment_17 * 0x3u + matrix_accumulator_2;
    }

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    if (!condition_2) {
        matrix_accumulator_2 = matrix_fragment_17 * 0x2u + matrix_fragment_4;

        matrix_fragment_17 = matrix_fragment_17 * 0x3u + matrix_fragment_4;
    }

    if (!condition_0) {
        matrix_fragment_15 = packed_half_0 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_22, matrix_fragment_23, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    if (!condition_2) {
        matrix_fragment_19 = packed_half_0 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    if (!condition_0) {
        matrix_fragment_27 = packed_half_0 * 0x4u;
    }

    if (!condition_2) {
        packed_half_0 = packed_half_0 * 0x4u;
    }

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    if (!condition_0) {
        matrix_fragment_27 = matrix_fragment_16 * uniform_index_word_5 + matrix_fragment_27;
    }

    if (!condition_2) {
        matrix_fragment_33 = matrix_fragment_17 * uniform_index_word_5 + packed_half_0;
    }

    matrix_fragment_16 = parameters[parameter_output_downsample_optional_low_16];
    matrix_fragment_17 = parameters[parameter_output_downsample_optional_high_17];

    if (!condition_2) {
        matrix_fragment_5 = matrix_fragment_4 * uniform_index_word_5 + matrix_fragment_5;

        matrix_fragment_4 = matrix_fragment_4 + uniform_index_word_4;
    }

    matrix_fragment_6 = matrix_fragment_6 & 0xfffffffcu;

    if (!condition_0) {
        matrix_fragment_15 = matrix_fragment_14 * uniform_index_word_5 + matrix_fragment_15;
    }

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                  select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_19 = matrix_accumulator_2 * uniform_index_word_5 + matrix_fragment_19;
    }

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_13 = matrix_fragment_4 * uniform_index_word_5 + matrix_fragment_13;
    }

    if (!condition_0) {
        matrix_accumulator_3 = matrix_accumulator_3 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_5 = matrix_fragment_5 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    if (!condition_0) {
        matrix_fragment_7 = matrix_fragment_7 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    condition_1 = combine_words(matrix_fragment_16, matrix_fragment_17) != 0ull;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_13 = matrix_fragment_13 + matrix_fragment_12 + (0u - matrix_fragment_6);

        matrix_fragment_23 = permute_signed(packed_half_0, matrix_fragment_23, 0x5410u);
    }

    if (!condition_0) {
        matrix_fragment_15 = matrix_fragment_15 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                  select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_35 = permute_signed(matrix_fragment_35, matrix_accumulator_2, 0x5410u);
    }

    if (!condition_0) {
        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(matrix_accumulator_3)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (!condition_2) {
        matrix_fragment_19 = matrix_fragment_19 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    if (!condition_0) {
        matrix_fragment_21 = permute_signed(matrix_fragment_20, matrix_fragment_21, 0x5410u);
    }

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_39 = permute_signed(matrix_fragment_4, matrix_fragment_39, 0x5410u);

        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (!condition_0) {
        matrix_fragment_27 = matrix_fragment_27 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    if (!condition_2) {
        matrix_fragment_33 = matrix_fragment_33 + matrix_fragment_12 + (0u - matrix_fragment_6);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_6, matrix_fragment_7,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        matrix_fragment_25 = permute_signed(matrix_fragment_8, matrix_fragment_25, 0x5410u);
    }

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    if (!condition_2) {
        matrix_fragment_29 = permute_signed(packed_half_0, matrix_fragment_11, 0x5410u);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_10, matrix_fragment_11,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        matrix_fragment_31 = permute_signed(matrix_fragment_34, matrix_fragment_31, 0x5410u);

        pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3),
                                    matrix_fragment_21, memory_ranges, error_counts);

        matrix_fragment_37 = permute_signed(matrix_fragment_28, matrix_fragment_37, 0x5410u);
    }

    if (!condition_2) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(matrix_fragment_19)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_4, matrix_fragment_5),
                                    matrix_fragment_23, memory_ranges, error_counts);
    }

    if (!condition_0) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    uniform_index_word_10 = uniform_index_word_10 + 0x1u;

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_6, matrix_fragment_7),
                                    matrix_fragment_25, memory_ranges, error_counts);
    }

    uniform_index_word_11 = uniform_index_word_11 + 0x1u;

    if (!condition_2) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_fragment_33)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        pipeline_write<CheckBounds>(combine_words(matrix_fragment_8, matrix_fragment_9),
                                    matrix_fragment_29, memory_ranges, error_counts);
    }

    uniform_index_word_10 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x1u >> 32) + uniform_index_word_10;

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_10, matrix_fragment_11),
                                    matrix_fragment_31, memory_ranges, error_counts);
    }

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x1u >> 32) + uniform_index_word_11;

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_12, matrix_fragment_13),
                                    matrix_fragment_35, memory_ranges, error_counts);
    }

    if (!condition_0) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_14, matrix_fragment_15),
                                    matrix_fragment_37, memory_ranges, error_counts);
    }

    if (!condition_2) {
        pipeline_write<CheckBounds>(combine_words(matrix_fragment_16, matrix_fragment_17),
                                    matrix_fragment_39, memory_ranges, error_counts);
    }

    if (!condition_1) {
        return;
    }

    uniform_index_word_4 = parameters[parameter_tensor_height_18];
    uniform_index_word_5 = parameters[parameter_tensor_width_19];

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_10) >> 0x1u);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x1u);

    matrix_fragment_5 = uniform_index_word_5;

    matrix_fragment_15 = uniform_index_word_4;

    condition_0 = int32_t(matrix_fragment_5) <= int32_t(uniform_index_word_11);

    condition_1 = int32_t(matrix_fragment_15) <= int32_t(uniform_index_word_10);

    if (condition_0) {
        return;
    }

    uniform_index_word_12 = blockDim.x;

    matrix_fragment_7 = threadIdx.z;

    packed_half_0 = matrix_fragment_5 * 0x4u;

    uniform_index_word_13 = blockDim.y;

    matrix_fragment_14 = blockDim.z;

    matrix_fragment_10 = threadIdx.y;

    uniform_index_word_5 = uniform_index_word_6 + 0x4u;

    uniform_address_word_9 = uniform_index_word_7 + 0x4u;

    matrix_fragment_12 = threadIdx.x;

    uniform_condition_0 = int32_t(uniform_index_word_5) > int32_t(uniform_index_word_10);

    matrix_accumulator_2 = gridDim.y;

    uniform_index_word_4 = gridDim.x;

    uniform_condition_1 = int32_t(uniform_address_word_9) > int32_t(uniform_index_word_11);

    condition_0 = uniform_condition_0;

    uniform_index_word_5 = uniform_index_word_12 * uniform_index_word_13;

    condition_1 = uniform_condition_1;

    condition_0 = int32_t(matrix_fragment_15) > int32_t(uniform_index_word_6) && condition_0;

    condition_1 = int32_t(matrix_fragment_5) > int32_t(uniform_index_word_7) && condition_1;

    matrix_accumulator_3 = matrix_fragment_14 * uniform_index_word_5;

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_index_word_8;

    condition_0 = condition_0 | condition_1;

    uniform_index_word_4 = uniform_index_word_4 + (0u - 0x8u);

    matrix_accumulator_2 = (matrix_accumulator_2 << 0x3u) + uniform_parameter_word_15;

    matrix_fragment_4 = matrix_fragment_7 * uniform_index_word_13 + matrix_fragment_10;

    uniform_index_word_4 =
        uint32_t(combine_words(uniform_index_word_4, 0u) << 0x1u >> 32) + uniform_index_word_4;

    matrix_accumulator_2 = matrix_accumulator_2 + (0u - 0x8u);

    matrix_fragment_4 = matrix_fragment_4 * uniform_index_word_12 + matrix_fragment_12;

    uniform_index_word_4 =
        uint32_t(int64_t(int32_t(uniform_index_word_4)) * (1ll << 0x1fu) >> 32) + 0x4u;

    matrix_accumulator_2 =
        uint32_t(combine_words(matrix_accumulator_2, 0u) << 0x1u >> 32) + matrix_accumulator_2;

    matrix_fragment_6 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1fu) >> 32) + 0x4u;

    matrix_accumulator_2 = packed_half_0 * matrix_fragment_15;

    matrix_fragment_6 = min(int32_t(matrix_fragment_6), int32_t(matrix_fragment_15));
    if (condition_0) {
        __syncthreads();

        condition_0 = uint32_t(matrix_fragment_4) > uint32_t(0x7fu);

        uniform_index_word_8 = parameters[parameter_output_downsample_optional_low_16];
        uniform_address_word_9 = parameters[parameter_output_downsample_optional_high_17];

        uniform_index_word_20 = parameters[parameter_output_downsample_optional_low_16];
        uniform_address_word_21 = parameters[parameter_output_downsample_optional_high_17];
        if (!condition_0) {
            const uint32_t next_boundary_index =
                ((matrix_fragment_7 + matrix_fragment_14) * uniform_index_word_13 + matrix_fragment_10) *
                    uniform_index_word_12 + matrix_fragment_12;
            matrix_fragment_12 = matrix_fragment_4;
            matrix_fragment_7 = boundary_tail_steps<false>(next_boundary_index, 128u, matrix_accumulator_3);

            matrix_fragment_8 = matrix_fragment_7 + 0x1u;

            matrix_fragment_8 = matrix_fragment_8 & 0x3u;
            condition_0 = matrix_fragment_8 != 0;
            if (condition_0) {
                matrix_fragment_16 = parameters[parameter_tensor_height_18];
                matrix_fragment_17 = parameters[parameter_tensor_width_19];

                matrix_fragment_13 = 0u - matrix_fragment_8;

                matrix_fragment_12 = matrix_fragment_4;
                do {
                    matrix_fragment_5 = uint32_t(uint32_t(matrix_fragment_12) >> 0x3u);

                    matrix_fragment_11 =
                        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x1bu >> 32) +
                        uniform_index_word_6;

                    matrix_fragment_5 = matrix_fragment_5 & 0x3u;

                    matrix_fragment_15 = matrix_fragment_16;

                    matrix_fragment_13 = matrix_fragment_13 + 0x1u;

                    matrix_fragment_10 = matrix_fragment_5 + uniform_index_word_7;

                    matrix_fragment_5 = matrix_fragment_17;

                    condition_0 = int32_t(matrix_fragment_11) >= int32_t(matrix_fragment_15);

                    condition_2 = int32_t(matrix_fragment_10) >= int32_t(uniform_index_word_11);

                    condition_1 = int32_t(matrix_fragment_10) >= int32_t(matrix_fragment_5);

                    condition_0 = int32_t(matrix_fragment_11) < int32_t(0u) || condition_0;

                    condition_2 = int32_t(matrix_fragment_11) < int32_t(uniform_index_word_10) &&
                                  !condition_2;

                    condition_1 = int32_t(matrix_fragment_10) < int32_t(0u) || condition_1;

                    condition_0 = condition_2 | condition_0 | condition_1;

                    condition_1 = uint32_t(matrix_fragment_13) != 0u;
                    if (!condition_0) {
                        matrix_fragment_9 = matrix_fragment_12 & 0x7u;

                        matrix_fragment_10 = matrix_fragment_10 * 0x4u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(matrix_accumulator_2)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(packed_half_0)) * int64_t(int32_t(matrix_fragment_11)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        matrix_fragment_11 = 0u;

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

                        const uint64_t byte_address_0 = scaled_word_address(
                            combine_words(uniform_index_word_8, uniform_address_word_9),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        matrix_fragment_10 = uint32_t(byte_address_0);

                        matrix_fragment_11 = uint32_t(byte_address_0 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    matrix_fragment_12 = matrix_accumulator_3 + matrix_fragment_12;
                } while (condition_1);
            }
            condition_0 = uint32_t(matrix_fragment_7) >= uint32_t(0x3u);
            if (condition_0) {
                matrix_fragment_28 = parameters[parameter_tensor_height_18];
                matrix_fragment_29 = parameters[parameter_tensor_width_19];

                matrix_fragment_5 = matrix_fragment_14 * uniform_index_word_13;

                matrix_fragment_18 = matrix_accumulator_3 + matrix_fragment_12;

                matrix_fragment_7 = matrix_fragment_5 * uniform_index_word_12;

                matrix_fragment_14 = matrix_fragment_7 * 0x2u + matrix_fragment_12;

                matrix_fragment_16 = matrix_fragment_7 * 0x3u + matrix_fragment_12;
                do {
                    matrix_fragment_8 = uint32_t(uint32_t(matrix_fragment_12) >> 0x3u);

                    matrix_fragment_13 =
                        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x1bu >> 32) +
                        uniform_index_word_6;

                    matrix_fragment_8 = matrix_fragment_8 & 0x3u;

                    matrix_fragment_15 = matrix_fragment_28;

                    matrix_fragment_5 = matrix_fragment_29;

                    matrix_fragment_9 = uint32_t(uint32_t(matrix_fragment_18) >> 0x3u);

                    matrix_fragment_20 = matrix_fragment_8 + uniform_index_word_7;

                    matrix_fragment_10 = uint32_t(uint32_t(matrix_fragment_14) >> 0x3u);

                    condition_2 = int32_t(matrix_fragment_13) >= int32_t(matrix_fragment_15);

                    matrix_fragment_9 = matrix_fragment_9 & 0x3u;

                    condition_3 = int32_t(matrix_fragment_20) >= int32_t(uniform_index_word_11);

                    condition_4 = int32_t(matrix_fragment_20) >= int32_t(matrix_fragment_5);

                    matrix_fragment_10 = matrix_fragment_10 & 0x3u;

                    condition_2 = int32_t(matrix_fragment_13) < int32_t(0u) || condition_2;

                    condition_3 = int32_t(matrix_fragment_13) < int32_t(uniform_index_word_10) &&
                                  !condition_3;

                    condition_4 = int32_t(matrix_fragment_20) < int32_t(0u) || condition_4;

                    matrix_fragment_17 =
                        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x1bu >> 32) +
                        uniform_index_word_6;

                    matrix_fragment_19 =
                        uint32_t(combine_words(matrix_fragment_14, 0u) << 0x1bu >> 32) +
                        uniform_index_word_6;

                    matrix_fragment_22 = matrix_fragment_9 + uniform_index_word_7;

                    matrix_fragment_24 = matrix_fragment_10 + uniform_index_word_7;

                    matrix_fragment_11 = uint32_t(uint32_t(matrix_fragment_16) >> 0x3u);

                    condition_2 = condition_3 | condition_2 | condition_4;

                    condition_0 = int32_t(matrix_fragment_17) >= int32_t(matrix_fragment_15);

                    condition_1 = int32_t(matrix_fragment_19) >= int32_t(matrix_fragment_15);

                    condition_5 = int32_t(matrix_fragment_22) >= int32_t(matrix_fragment_5);

                    condition_6 = int32_t(matrix_fragment_22) >= int32_t(uniform_index_word_11);

                    condition_3 = int32_t(matrix_fragment_24) >= int32_t(matrix_fragment_5);

                    condition_4 = int32_t(matrix_fragment_24) >= int32_t(uniform_index_word_11);

                    matrix_fragment_11 = matrix_fragment_11 & 0x3u;

                    condition_0 = int32_t(matrix_fragment_17) < int32_t(0u) || condition_0;

                    condition_1 = int32_t(matrix_fragment_19) < int32_t(0u) || condition_1;

                    condition_5 = int32_t(matrix_fragment_22) < int32_t(0u) || condition_5;

                    condition_6 = int32_t(matrix_fragment_17) < int32_t(uniform_index_word_10) &&
                                  !condition_6;

                    condition_3 = int32_t(matrix_fragment_24) < int32_t(0u) || condition_3;

                    condition_4 = int32_t(matrix_fragment_19) < int32_t(uniform_index_word_10) &&
                                  !condition_4;

                    matrix_fragment_21 =
                        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x1bu >> 32) +
                        uniform_index_word_6;

                    matrix_fragment_26 = matrix_fragment_11 + uniform_index_word_7;

                    condition_0 = condition_6 | condition_0 | condition_5;

                    condition_1 = condition_4 | condition_1 | condition_3;

                    condition_5 = int32_t(matrix_fragment_21) >= int32_t(matrix_fragment_15);

                    condition_6 = int32_t(matrix_fragment_26) >= int32_t(matrix_fragment_5);

                    condition_3 = int32_t(matrix_fragment_26) >= int32_t(uniform_index_word_11);

                    condition_5 = int32_t(matrix_fragment_21) < int32_t(0u) || condition_5;

                    condition_6 = int32_t(matrix_fragment_26) < int32_t(0u) || condition_6;

                    condition_3 = int32_t(matrix_fragment_21) < int32_t(uniform_index_word_10) &&
                                  !condition_3;

                    condition_3 = condition_3 | condition_5 | condition_6;
                    if (!condition_2) {
                        matrix_fragment_9 = matrix_fragment_12 & 0x7u;

                        matrix_fragment_10 = matrix_fragment_20 * 0x4u;

                        matrix_fragment_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(matrix_accumulator_2)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(packed_half_0)) * int64_t(int32_t(matrix_fragment_13)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

                        const uint64_t byte_address_1 = scaled_word_address(
                            combine_words(uniform_index_word_20, uniform_address_word_21),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        matrix_fragment_10 = uint32_t(byte_address_1);

                        matrix_fragment_11 = uint32_t(byte_address_1 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    if (!condition_0) {
                        matrix_fragment_9 = matrix_fragment_18 & 0x7u;

                        matrix_fragment_10 = matrix_fragment_22 * 0x4u;

                        matrix_fragment_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(matrix_accumulator_2)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(packed_half_0)) * int64_t(int32_t(matrix_fragment_17)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

                        const uint64_t byte_address_2 = scaled_word_address(
                            combine_words(uniform_index_word_20, uniform_address_word_21),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        matrix_fragment_10 = uint32_t(byte_address_2);

                        matrix_fragment_11 = uint32_t(byte_address_2 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    if (!condition_1) {
                        matrix_fragment_9 = matrix_fragment_14 & 0x7u;

                        matrix_fragment_10 = matrix_fragment_24 * 0x4u;

                        matrix_fragment_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(matrix_accumulator_2)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(packed_half_0)) * int64_t(int32_t(matrix_fragment_19)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

                        const uint64_t byte_address_3 = scaled_word_address(
                            combine_words(uniform_index_word_20, uniform_address_word_21),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        matrix_fragment_10 = uint32_t(byte_address_3);

                        matrix_fragment_11 = uint32_t(byte_address_3 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    matrix_fragment_12 =
                        matrix_accumulator_3 + matrix_fragment_12 + matrix_accumulator_3;
                    if (!condition_3) {
                        matrix_fragment_9 = matrix_fragment_16 & 0x7u;

                        matrix_fragment_10 = matrix_fragment_26 * 0x4u;

                        matrix_fragment_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(matrix_accumulator_2)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(packed_half_0)) * int64_t(int32_t(matrix_fragment_21)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                combine_words(matrix_fragment_10, matrix_fragment_11));

                        const uint64_t byte_address_4 = scaled_word_address(
                            combine_words(uniform_index_word_20, uniform_address_word_21),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        matrix_fragment_10 = uint32_t(byte_address_4);

                        matrix_fragment_11 = uint32_t(byte_address_4 >> 32);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11), 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x4ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0x8ull, 0u,
                            memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(
                            combine_words(matrix_fragment_10, matrix_fragment_11) + 0xcull, 0u,
                            memory_ranges, error_counts);
                    }
                    matrix_fragment_12 =
                        matrix_accumulator_3 + matrix_fragment_12 + matrix_accumulator_3;

                    matrix_fragment_14 = matrix_fragment_7 * 0x4u + matrix_fragment_14;

                    matrix_fragment_16 = matrix_fragment_7 * 0x4u + matrix_fragment_16;

                    condition_0 = uint32_t(matrix_fragment_12) >= uint32_t(0x80u);

                    matrix_fragment_18 = matrix_fragment_7 * 0x4u + matrix_fragment_18;
                } while (!condition_0);
            }
        }
    }
    uniform_index_word_5 = blockIdx.z;

    {
        uint32_t result_word;
        result_word = uniform_address_word_14 | uniform_block_index_18 | uniform_index_word_5;
        uniform_condition_0 = result_word != 0;
    }

    condition_0 = uniform_condition_0;

    if (condition_0) {
        return;
    }

    matrix_fragment_7 = max(int32_t(matrix_fragment_6), int32_t(uniform_index_word_10));

    matrix_fragment_8 = min(int32_t(matrix_fragment_5), int32_t(uniform_index_word_4));

    uniform_index_word_5 = parameters[parameter_tensor_width_19];

    matrix_fragment_10 = 0u - matrix_fragment_7 + matrix_fragment_15;

    uniform_index_word_12 = parameters[parameter_tensor_width_19];

    matrix_fragment_8 = max(int32_t(matrix_fragment_8), int32_t(uniform_index_word_11));

    condition_0 = int32_t(matrix_fragment_10) >= int32_t(0x1u);

    uniform_index_word_6 = parameters[parameter_output_downsample_optional_low_16];
    uniform_index_word_7 = parameters[parameter_output_downsample_optional_high_17];

    uniform_index_word_8 = parameters[parameter_output_downsample_optional_low_16];
    uniform_address_word_9 = parameters[parameter_output_downsample_optional_high_17];
    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_10 * matrix_fragment_5;

        matrix_fragment_9 = matrix_fragment_9 * 0x8u;

        condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_9);
        if (!condition_0) {
            matrix_fragment_16 = matrix_fragment_4;
            matrix_fragment_11 = boundary_tail_steps<true>(
                matrix_accumulator_3 + matrix_fragment_4, matrix_fragment_9, matrix_accumulator_3);

            matrix_fragment_10 = matrix_fragment_11 + 0x1u;

            condition_0 = uint32_t(matrix_fragment_11) >= uint32_t(0x3u);

            matrix_fragment_14 = matrix_fragment_10 & 0x3u;
            condition_1 = matrix_fragment_14 != 0;
            if (condition_1) {
                matrix_fragment_18 =
                    signed_word_magnitude(matrix_fragment_5);

                condition_2 = int32_t(matrix_fragment_5) != int32_t(0u);

                matrix_fragment_14 = 0u - matrix_fragment_14;

                matrix_fragment_12 = __float_as_uint(__int2float_ru(matrix_fragment_18));

                matrix_fragment_17 = ~matrix_fragment_5;

                matrix_fragment_16 = matrix_fragment_4;

                matrix_fragment_12 = reciprocal_bits(matrix_fragment_12);

                matrix_fragment_10 = matrix_fragment_12 + 0xffffffeu;

                matrix_fragment_11 = __float2uint_rz(__uint_as_float(matrix_fragment_10));

                matrix_fragment_10 = 0u;

                matrix_fragment_15 = 0u - matrix_fragment_11;

                matrix_fragment_15 = matrix_fragment_15 * matrix_fragment_18;

                matrix_fragment_15 =
                    uint32_t(uint64_t(matrix_fragment_11) * matrix_fragment_15 +
                                 combine_words(matrix_fragment_10, matrix_fragment_11) >>
                             32);
                do {
                    matrix_fragment_20 = uint32_t(uint32_t(matrix_fragment_16) >> 0x3u);

                    matrix_fragment_5 = uniform_index_word_5;

                    matrix_fragment_14 = matrix_fragment_14 + 0x1u;

                    matrix_fragment_12 = int32_t(matrix_fragment_20) < 0 ? 0u - matrix_fragment_20
                                                                         : matrix_fragment_20;

                    matrix_fragment_22 =
                        signed_word_magnitude(matrix_fragment_5);

                    matrix_fragment_10 =
                        uint32_t(uint64_t(matrix_fragment_15) * matrix_fragment_12 >> 32);

                    matrix_fragment_11 = 0u - matrix_fragment_10;

                    matrix_fragment_11 =
                        matrix_fragment_22 * matrix_fragment_11 + matrix_fragment_12;

                    condition_3 = uint32_t(matrix_fragment_18) > uint32_t(matrix_fragment_11);

                    if (!condition_3) {
                        matrix_fragment_11 = matrix_fragment_11 + (0u - matrix_fragment_22);

                        matrix_fragment_10 = matrix_fragment_10 + 0x1u;
                    }

                    condition_1 = uint32_t(matrix_fragment_11) >= uint32_t(matrix_fragment_18);

                    matrix_fragment_11 = matrix_fragment_20 ^ matrix_fragment_5;

                    condition_3 = int32_t(matrix_fragment_11) >= int32_t(0u);

                    matrix_fragment_11 = matrix_fragment_16 & 0x7u;

                    matrix_fragment_16 = matrix_accumulator_3 + matrix_fragment_16;

                    if (condition_1) {
                        matrix_fragment_10 = matrix_fragment_10 + 0x1u;
                    }

                    if (!condition_3) {
                        matrix_fragment_10 = 0u - matrix_fragment_10;
                    }

                    matrix_fragment_12 = !condition_2 ? matrix_fragment_17 : matrix_fragment_10;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        int64_t(int32_t(matrix_fragment_11)) *
                                                int64_t(int32_t(matrix_accumulator_2)) +
                                            0ull);

                    matrix_fragment_13 = matrix_fragment_7 + matrix_fragment_12;

                    matrix_fragment_12 = 0u - matrix_fragment_12;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        int64_t(int32_t(packed_half_0)) *
                                                int64_t(int32_t(matrix_fragment_13)) +
                                            combine_words(matrix_fragment_10, matrix_fragment_11));

                    matrix_fragment_13 =
                        matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_20;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        uint64_t(uint32_t(matrix_fragment_13)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_10, matrix_fragment_11));

                    const uint64_t byte_address_5 = scaled_word_address(
                        combine_words(uniform_index_word_6, uniform_index_word_7),
                        combine_words(matrix_fragment_10, matrix_fragment_11));
                    matrix_fragment_12 = uint32_t(byte_address_5);

                    matrix_fragment_13 = uint32_t(byte_address_5 >> 32);

                    condition_1 = uint32_t(matrix_fragment_14) != 0u;

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
                } while (condition_1);
            }
            if (condition_0) {
                matrix_fragment_25 =
                    signed_word_magnitude(matrix_fragment_5);

                matrix_fragment_5 = __float_as_uint(__int2float_ru(matrix_fragment_25));

                matrix_fragment_5 = reciprocal_bits(matrix_fragment_5);

                matrix_fragment_10 = matrix_fragment_5 + 0xffffffeu;

                matrix_fragment_11 = __float2uint_rz(__uint_as_float(matrix_fragment_10));

                matrix_fragment_10 = 0u;

                matrix_fragment_12 = 0u - matrix_fragment_11;

                matrix_fragment_13 = matrix_fragment_12 * matrix_fragment_25;

                matrix_fragment_27 =
                    uint32_t(uint64_t(matrix_fragment_11) * matrix_fragment_13 +
                                 combine_words(matrix_fragment_10, matrix_fragment_11) >>
                             32);
                do {
                    matrix_fragment_5 = uniform_index_word_12;

                    matrix_fragment_22 = uint32_t(uint32_t(matrix_fragment_16) >> 0x3u);

                    matrix_fragment_24 = matrix_accumulator_3 + matrix_fragment_16;

                    matrix_fragment_21 =
                        signed_word_magnitude(matrix_fragment_5);

                    matrix_fragment_13 = int32_t(matrix_fragment_22) < 0 ? 0u - matrix_fragment_22
                                                                         : matrix_fragment_22;

                    matrix_fragment_14 = __float_as_uint(__int2float_ru(matrix_fragment_21));

                    matrix_fragment_26 = uint32_t(uint32_t(matrix_fragment_24) >> 0x3u);

                    matrix_fragment_12 =
                        uint32_t(uint64_t(matrix_fragment_27) * matrix_fragment_13 >> 32);

                    matrix_fragment_28 = matrix_accumulator_3 + matrix_fragment_24;

                    matrix_fragment_15 = int32_t(matrix_fragment_26) < 0 ? 0u - matrix_fragment_26
                                                                         : matrix_fragment_26;

                    matrix_fragment_29 = ~matrix_fragment_5;

                    matrix_fragment_10 = 0u - matrix_fragment_12;

                    matrix_fragment_32 = matrix_accumulator_3 + matrix_fragment_28;

                    matrix_fragment_30 = uint32_t(uint32_t(matrix_fragment_28) >> 0x3u);

                    matrix_fragment_13 =
                        matrix_fragment_21 * matrix_fragment_10 + matrix_fragment_13;

                    matrix_fragment_14 = reciprocal_bits(matrix_fragment_14);

                    matrix_fragment_34 = uint32_t(uint32_t(matrix_fragment_32) >> 0x3u);

                    matrix_fragment_18 = int32_t(matrix_fragment_30) < 0 ? 0u - matrix_fragment_30
                                                                         : matrix_fragment_30;

                    condition_1 = uint32_t(matrix_fragment_25) > uint32_t(matrix_fragment_13);

                    matrix_fragment_20 = int32_t(matrix_fragment_34) < 0 ? 0u - matrix_fragment_34
                                                                         : matrix_fragment_34;

                    matrix_fragment_23 = matrix_fragment_32 & 0x7u;

                    matrix_fragment_11 = matrix_fragment_14 + 0xffffffeu;

                    if (!condition_1) {
                        matrix_fragment_13 = matrix_fragment_13 + (0u - matrix_fragment_21);

                        matrix_fragment_12 = matrix_fragment_12 + 0x1u;
                    }

                    matrix_fragment_11 = __float2uint_rz(__uint_as_float(matrix_fragment_11));

                    condition_0 = uint32_t(matrix_fragment_13) >= uint32_t(matrix_fragment_25);

                    matrix_fragment_25 = matrix_fragment_21;

                    if (condition_0) {
                        matrix_fragment_12 = matrix_fragment_12 + 0x1u;
                    }

                    matrix_fragment_10 = 0u - matrix_fragment_11;

                    condition_0 = int32_t(matrix_fragment_5) != int32_t(0u);

                    matrix_fragment_27 = matrix_fragment_10 * matrix_fragment_21;

                    matrix_fragment_10 = 0u;

                    matrix_fragment_27 =
                        uint32_t(uint64_t(matrix_fragment_11) * matrix_fragment_27 +
                                     combine_words(matrix_fragment_10, matrix_fragment_11) >>
                                 32);

                    matrix_fragment_11 = matrix_fragment_22 ^ matrix_fragment_5;

                    matrix_fragment_14 =
                        uint32_t(uint64_t(matrix_fragment_27) * matrix_fragment_15 >> 32);

                    condition_3 = int32_t(matrix_fragment_11) >= int32_t(0u);

                    matrix_fragment_17 =
                        uint32_t(uint64_t(matrix_fragment_27) * matrix_fragment_18 >> 32);

                    matrix_fragment_10 = 0u - matrix_fragment_14;

                    matrix_fragment_19 =
                        uint32_t(uint64_t(matrix_fragment_27) * matrix_fragment_20 >> 32);

                    matrix_fragment_11 = 0u - matrix_fragment_17;

                    matrix_fragment_15 =
                        matrix_fragment_21 * matrix_fragment_10 + matrix_fragment_15;

                    matrix_fragment_13 = 0u - matrix_fragment_19;

                    matrix_fragment_18 =
                        matrix_fragment_21 * matrix_fragment_11 + matrix_fragment_18;

                    if (!condition_3) {
                        matrix_fragment_12 = 0u - matrix_fragment_12;
                    }

                    matrix_fragment_11 = matrix_fragment_16 & 0x7u;

                    condition_2 = uint32_t(matrix_fragment_25) > uint32_t(matrix_fragment_15);

                    matrix_fragment_20 =
                        matrix_fragment_21 * matrix_fragment_13 + matrix_fragment_20;

                    condition_5 = uint32_t(matrix_fragment_25) > uint32_t(matrix_fragment_18);

                    matrix_fragment_12 = !condition_0 ? matrix_fragment_29 : matrix_fragment_12;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        int64_t(int32_t(matrix_fragment_11)) *
                                                int64_t(int32_t(matrix_accumulator_2)) +
                                            0ull);

                    matrix_fragment_16 = matrix_fragment_26 ^ matrix_fragment_5;

                    condition_1 = uint32_t(matrix_fragment_25) > uint32_t(matrix_fragment_20);

                    matrix_fragment_13 = matrix_fragment_7 + matrix_fragment_12;

                    matrix_fragment_12 = 0u - matrix_fragment_12;

                    condition_3 = int32_t(matrix_fragment_16) >= int32_t(0u);

                    if (!condition_2) {
                        matrix_fragment_15 = matrix_fragment_15 + (0u - matrix_fragment_21);
                    }

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        int64_t(int32_t(packed_half_0)) *
                                                int64_t(int32_t(matrix_fragment_13)) +
                                            combine_words(matrix_fragment_10, matrix_fragment_11));

                    if (!condition_5) {
                        matrix_fragment_18 = matrix_fragment_18 + (0u - matrix_fragment_21);
                    }

                    if (!condition_2) {
                        matrix_fragment_14 = matrix_fragment_14 + 0x1u;
                    }

                    if (!condition_5) {
                        matrix_fragment_17 = matrix_fragment_17 + 0x1u;
                    }

                    condition_6 = uint32_t(matrix_fragment_15) >= uint32_t(matrix_fragment_25);

                    matrix_fragment_15 = matrix_fragment_30 ^ matrix_fragment_5;

                    matrix_fragment_13 =
                        matrix_fragment_5 * matrix_fragment_12 + matrix_fragment_22;

                    if (!condition_1) {
                        matrix_fragment_20 = matrix_fragment_20 + (0u - matrix_fragment_21);
                    }

                    condition_4 = uint32_t(matrix_fragment_18) >= uint32_t(matrix_fragment_25);

                    matrix_fragment_12 = matrix_fragment_34 ^ matrix_fragment_5;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        uint64_t(uint32_t(matrix_fragment_13)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_10, matrix_fragment_11));

                    if (!condition_1) {
                        matrix_fragment_19 = matrix_fragment_19 + 0x1u;
                    }

                    condition_2 = uint32_t(matrix_fragment_20) >= uint32_t(matrix_fragment_25);

                    matrix_fragment_21 = matrix_fragment_28 & 0x7u;

                    condition_5 = int32_t(matrix_fragment_12) >= int32_t(0u);

                    const uint64_t byte_address_6 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(matrix_fragment_10, matrix_fragment_11));
                    matrix_fragment_12 = uint32_t(byte_address_6);

                    if (condition_6) {
                        matrix_fragment_14 = matrix_fragment_14 + 0x1u;
                    }

                    condition_6 = int32_t(matrix_fragment_15) >= int32_t(0u);

                    matrix_fragment_13 = uint32_t(byte_address_6 >> 32);

                    if (condition_4) {
                        matrix_fragment_17 = matrix_fragment_17 + 0x1u;
                    }

                    if (!condition_3) {
                        matrix_fragment_14 = 0u - matrix_fragment_14;
                    }

                    matrix_fragment_11 = matrix_fragment_24 & 0x7u;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13), 0u, memory_ranges,
                        error_counts);

                    if (condition_2) {
                        matrix_fragment_19 = matrix_fragment_19 + 0x1u;
                    }

                    matrix_fragment_14 = !condition_0 ? matrix_fragment_29 : matrix_fragment_14;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        int64_t(int32_t(matrix_fragment_11)) *
                                                int64_t(int32_t(matrix_accumulator_2)) +
                                            0ull);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    if (!condition_5) {
                        matrix_fragment_19 = 0u - matrix_fragment_19;
                    }

                    if (!condition_6) {
                        matrix_fragment_17 = 0u - matrix_fragment_17;
                    }

                    matrix_fragment_15 = matrix_fragment_7 + matrix_fragment_14;

                    matrix_fragment_18 = 0u - matrix_fragment_14;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_20 = !condition_0 ? matrix_fragment_29 : matrix_fragment_19;

                    matrix_fragment_16 = !condition_0 ? matrix_fragment_29 : matrix_fragment_17;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        int64_t(int32_t(packed_half_0)) *
                                                int64_t(int32_t(matrix_fragment_15)) +
                                            combine_words(matrix_fragment_10, matrix_fragment_11));

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_17 =
                        matrix_fragment_5 * matrix_fragment_18 + matrix_fragment_26;

                    matrix_fragment_22 = 0u - matrix_fragment_16;

                    split_address_words(matrix_fragment_14, matrix_fragment_15,
                                        int64_t(int32_t(matrix_fragment_21)) *
                                                int64_t(int32_t(matrix_accumulator_2)) +
                                            0ull);

                    matrix_fragment_21 = matrix_fragment_7 + matrix_fragment_16;

                    split_address_words(matrix_fragment_18, matrix_fragment_19,
                                        int64_t(int32_t(matrix_fragment_23)) *
                                                int64_t(int32_t(matrix_accumulator_2)) +
                                            0ull);

                    matrix_fragment_23 = matrix_fragment_7 + matrix_fragment_20;

                    matrix_fragment_20 = 0u - matrix_fragment_20;

                    split_address_words(matrix_fragment_10, matrix_fragment_11,
                                        uint64_t(uint32_t(matrix_fragment_17)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_10, matrix_fragment_11));

                    split_address_words(matrix_fragment_16, matrix_fragment_17,
                                        int64_t(int32_t(packed_half_0)) *
                                                int64_t(int32_t(matrix_fragment_21)) +
                                            combine_words(matrix_fragment_14, matrix_fragment_15));

                    const uint64_t byte_address_7 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(matrix_fragment_10, matrix_fragment_11));
                    matrix_fragment_14 = uint32_t(byte_address_7);

                    split_address_words(matrix_fragment_18, matrix_fragment_19,
                                        int64_t(int32_t(packed_half_0)) *
                                                int64_t(int32_t(matrix_fragment_23)) +
                                            combine_words(matrix_fragment_18, matrix_fragment_19));

                    matrix_fragment_15 =
                        matrix_fragment_5 * matrix_fragment_22 + matrix_fragment_30;

                    matrix_fragment_21 =
                        matrix_fragment_5 * matrix_fragment_20 + matrix_fragment_34;

                    split_address_words(matrix_fragment_16, matrix_fragment_17,
                                        uint64_t(uint32_t(matrix_fragment_15)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_16, matrix_fragment_17));

                    matrix_fragment_15 = uint32_t(byte_address_7 >> 32);

                    split_address_words(matrix_fragment_18, matrix_fragment_19,
                                        uint64_t(uint32_t(matrix_fragment_21)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(matrix_fragment_18, matrix_fragment_19));

                    const uint64_t byte_address_8 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(matrix_fragment_16, matrix_fragment_17));
                    matrix_fragment_10 = uint32_t(byte_address_8);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15), 0u, memory_ranges,
                        error_counts);

                    const uint64_t byte_address_9 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(matrix_fragment_18, matrix_fragment_19));
                    matrix_fragment_20 = uint32_t(byte_address_9);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_11 = uint32_t(byte_address_8 >> 32);

                    matrix_fragment_21 = uint32_t(byte_address_9 >> 32);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_16 = matrix_accumulator_3 + matrix_fragment_32;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    condition_0 = int32_t(matrix_fragment_16) >= int32_t(matrix_fragment_9);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_10, matrix_fragment_11), 0u, memory_ranges,
                        error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_10, matrix_fragment_11) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_20, matrix_fragment_21), 0u, memory_ranges,
                        error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_20, matrix_fragment_21) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_20, matrix_fragment_21) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_20, matrix_fragment_21) + 0xcull, 0u,
                        memory_ranges, error_counts);
                } while (!condition_0);
            }
        }
    }
    matrix_fragment_9 = 0u - matrix_fragment_8 + matrix_fragment_5;

    matrix_fragment_5 = min(int32_t(matrix_fragment_6), int32_t(matrix_fragment_9));

    condition_0 = int32_t(matrix_fragment_5) >= int32_t(0x1u);

    if (!condition_0) {
        return;
    }

    matrix_fragment_6 = matrix_fragment_6 * matrix_fragment_9;

    matrix_fragment_19 = matrix_fragment_6 * 0x8u;

    condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_19);

    if (condition_0) {
        return;
    }

    // Clear the final column-padding region only. Earlier tile/edge publication
    // and its synchronization stay here; see docs/BOUNDARY_KERNELS.md.
    const BoundaryVectorLayout padding_layout{
        combine_words(parameters[parameter_output_downsample_optional_low_16],
                      parameters[parameter_output_downsample_optional_high_17]),
        matrix_accumulator_2, // component stride, uint32 words
        packed_half_0, // row stride, uint32 words
        matrix_fragment_8}; // column origin, signed-word bits
    const uint32_t padding_width = matrix_fragment_9;
    const uint32_t first_padding_item = matrix_fragment_4;
    const uint32_t padding_extent = matrix_fragment_19;
    const uint32_t block_thread_count = matrix_accumulator_3;
    clear_boundary_rows<3>(padding_layout, padding_width, first_padding_item,
                           padding_extent, block_thread_count, [&](uint64_t address) {
        pipeline_write<CheckBounds>(address, 0u, memory_ranges, error_counts);
    });
}
extern "C" __global__ __launch_bounds__(
    32, 14) void phoenix_window_attention_channels_32_downsample(KernelArguments parameters) {
    run_window_attention_channels_32_downsample<false>(parameters.words, nullptr, nullptr);
}
