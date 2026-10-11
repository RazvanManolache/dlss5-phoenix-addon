// DLSS5 Phoenix optimized kernel: window_attention/channels_32/attention.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 4
#include "../../../common/device_memory_vector_half_permute.cuh"
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
    parameter_scalar_control_8 = 8,
    parameter_geometry_or_control_9 = 9,
    parameter_wait_optional_low_10 = 10,
    parameter_wait_optional_high_11 = 11,
    parameter_geometry_or_control_12 = 12,
    parameter_geometry_or_control_13 = 13,
    parameter_publication_optional_low_14 = 14,
    parameter_publication_optional_high_15 = 15,
};

struct alignas(8) KernelArguments {
    uint32_t words[24];
};
static_assert(sizeof(KernelArguments) == 96, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_32_attention(
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
    uint32_t packed_half_11 = 0;
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
    uint32_t matrix_accumulator_72 = 0;
    uint32_t matrix_accumulator_73 = 0;
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_accumulator_76 = 0;
    uint32_t matrix_accumulator_77 = 0;
    uint32_t matrix_accumulator_78 = 0;
    uint32_t matrix_accumulator_79 = 0;
    uint32_t matrix_accumulator_80 = 0;
    uint32_t matrix_accumulator_81 = 0;
    uint32_t matrix_accumulator_82 = 0;
    uint32_t matrix_accumulator_83 = 0;
    uint32_t matrix_fragment_84 = 0;
    uint32_t matrix_fragment_85 = 0;
    uint32_t matrix_fragment_86 = 0;
    uint32_t matrix_fragment_87 = 0;
    uint32_t matrix_accumulator_88 = 0;
    uint32_t matrix_accumulator_89 = 0;
    uint32_t matrix_accumulator_90 = 0;
    uint32_t packed_half_91 = 0;
    uint32_t matrix_accumulator_92 = 0;
    uint32_t matrix_accumulator_93 = 0;
    uint32_t matrix_accumulator_94 = 0;
    uint32_t matrix_accumulator_95 = 0;
    uint32_t matrix_accumulator_96 = 0;
    uint32_t matrix_accumulator_97 = 0;
    uint32_t packed_half_98 = 0;
    uint32_t packed_half_99 = 0;
    uint32_t packed_half_100 = 0;
    uint32_t packed_half_101 = 0;
    uint32_t matrix_accumulator_102 = 0;
    uint32_t matrix_accumulator_103 = 0;
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
    uint32_t packed_half_115 = 0;
    uint32_t matrix_accumulator_116 = 0;
    uint32_t matrix_accumulator_117 = 0;
    uint32_t packed_half_118 = 0;
    uint32_t packed_half_119 = 0;
    uint32_t matrix_accumulator_120 = 0;
    uint32_t matrix_accumulator_121 = 0;
    uint32_t matrix_accumulator_122 = 0;
    uint32_t matrix_accumulator_123 = 0;
    uint32_t matrix_accumulator_124 = 0;
    uint32_t matrix_accumulator_125 = 0;
    uint32_t matrix_accumulator_126 = 0;
    uint32_t matrix_accumulator_127 = 0;
    uint32_t packed_half_128 = 0;
    uint32_t packed_half_129 = 0;
    uint32_t matrix_accumulator_130 = 0;
    uint32_t matrix_accumulator_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t matrix_accumulator_133 = 0;
    uint32_t packed_half_134 = 0;
    uint32_t packed_half_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t packed_half_137 = 0;
    uint32_t matrix_accumulator_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t matrix_accumulator_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t packed_half_142 = 0;
    uint32_t packed_half_143 = 0;
    uint32_t packed_half_144 = 0;
    uint32_t packed_half_145 = 0;
    uint32_t matrix_accumulator_146 = 0;
    uint32_t matrix_accumulator_147 = 0;
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
    uint32_t packed_half_158 = 0;
    uint32_t packed_half_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t address_word_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t packed_half_165 = 0;
    uint32_t uniform_reused_word_4 = 0;
    uint32_t uniform_block_index_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_address_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_index_word_16 = 0;
    uint32_t uniform_parameter_word_17 = 0;
    uint32_t uniform_reused_word_18 = 0;
    uint32_t uniform_reused_word_19 = 0;
    uint32_t uniform_index_word_20 = 0;
    uint32_t uniform_index_word_21 = 0;
    uint32_t uniform_packed_half_22 = 0;
    uint32_t uniform_index_word_23 = 0;
    uint32_t uniform_index_word_24 = 0;
    uint32_t uniform_index_word_25 = 0;
    bool condition_0 = false;
    bool condition_1 = false;
    bool condition_2 = false;
    bool condition_3 = false;
    bool condition_4 = false;
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

    uniform_index_word_13 = parameters[parameter_geometry_or_control_13];

    uniform_index_word_20 = blockIdx.y;

    packed_half_0 = threadIdx.x;

    reused_word_1 = reused_word_1 + (0u - 0x8u);

    uniform_address_word_14 = parameters[parameter_geometry_or_control_12];

    packed_half_3 = threadIdx.y;

    uniform_address_word_15 = parameters[parameter_tensor_height_6];

    uniform_address_word_7 = blockIdx.x;

    packed_half_2 = threadIdx.z;

    uniform_index_word_11 = parameters[parameter_scalar_control_8];

    uniform_index_word_12 = parameters[parameter_tensor_width_7];

    uniform_index_word_8 = parameters[parameter_geometry_or_control_9];

    uniform_index_word_10 = min(int32_t(0u), int32_t(uniform_index_word_13));

    uniform_index_word_9 = min(int32_t(0u), int32_t(uniform_address_word_14));

    uniform_block_index_5 = 0u;

    uniform_reused_word_4 = 0u;

    uniform_index_word_10 = 0u - uniform_index_word_10 + uniform_address_word_15;

    uniform_reused_word_18 = 0u;
    uniform_reused_word_19 = 0u;

    uniform_address_word_6 = min(int32_t(0u), int32_t(uniform_index_word_11));

    uniform_index_word_9 = 0u - uniform_index_word_9 + uniform_index_word_12;

    uniform_condition_4 = uint32_t(uniform_index_word_10) != 0u;

    uniform_address_word_7 = (uniform_address_word_7 << 0x3u) + uniform_index_word_11;

    uniform_address_word_6 = 0u - uniform_address_word_6 + uniform_index_word_12;

    uniform_condition_3 = uint32_t(uniform_index_word_9) != 0u;

    uniform_index_word_8 = (uniform_index_word_20 << 0x3u) + uniform_index_word_8;

    uniform_address_word_14 = uniform_address_word_7 + (0u - uniform_address_word_14);

    uniform_condition_2 = uint32_t(uniform_address_word_6) != 0u;

    condition_0 = (packed_half_0 | packed_half_2 | packed_half_3) != 0;

    uniform_index_word_12 = uniform_index_word_8 + (0u - uniform_index_word_13);

    uniform_parameter_word_17 = uniform_address_word_14 + 0x8u;

    uniform_address_word_15 = uniform_address_word_14 + 0x7u;

    if (uniform_condition_4) {
        uniform_index_word_11 = uniform_index_word_10 + 0xffffffffu;
    }

    uniform_index_word_16 = uniform_index_word_12 + 0x8u;

    uniform_index_word_13 = uniform_index_word_12 + 0x7u;

    if (uniform_condition_3) {
        uniform_index_word_10 = uniform_index_word_9 + 0xffffffffu;
    }

    if (uniform_condition_4) {
        uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_11) >> 0x1fu);
    }

    uniform_condition_0 = int32_t(uniform_index_word_16) > int32_t(0u);

    if (uniform_condition_2) {
        uniform_address_word_6 = uniform_address_word_6 + 0xffffffffu;
    }

    uniform_index_word_13 =
        uint32_t(combine_words(uniform_index_word_13, 0u) << 0x1du >> 32) + 0x1u;

    if (uniform_condition_4) {
        uniform_index_word_11 =
            uint32_t(combine_words(uniform_index_word_9, 0u) << 0x3u >> 32) + uniform_index_word_11;
    }

    packed_half_0 = max(int32_t(0u), int32_t(uniform_index_word_12));

    if (uniform_condition_2) {
        uniform_index_word_9 = uint32_t(int32_t(uniform_address_word_6) >> 0x1fu);
    }

    uniform_index_word_13 = uniform_condition_0 ? uniform_index_word_13 : 0u;

    if (uniform_condition_4) {
        uniform_block_index_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_11)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (uniform_condition_3) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);
    }

    if (uniform_condition_2) {
        uniform_address_word_6 = uint32_t(combine_words(uniform_index_word_9, 0u) << 0x3u >> 32) +
                                 uniform_address_word_6;
    }

    packed_half_0 = uint32_t(uint32_t(packed_half_0) >> 0x3u);

    uniform_index_word_9 = min(int32_t(uniform_index_word_13), int32_t(uniform_block_index_5));

    uniform_address_word_15 =
        uint32_t(combine_words(uniform_address_word_15, 0u) << 0x1du >> 32) + 0x1u;

    uniform_condition_1 = int32_t(uniform_parameter_word_17) > int32_t(0u);

    if (uniform_condition_3) {
        uniform_index_word_11 = uint32_t(combine_words(uniform_index_word_11, 0u) << 0x3u >> 32) +
                                uniform_index_word_10;
    }

    if (uniform_condition_2) {
        uniform_address_word_6 = uint32_t(int32_t(uniform_address_word_6) >> 0x3u);
    }

    uniform_block_index_5 = 0u;

    condition_0 = int32_t(packed_half_0) >= int32_t(uniform_index_word_9) || condition_0;

    uniform_index_word_10 = uniform_condition_1 ? uniform_address_word_15 : 0u;

    if (uniform_condition_3) {
        uniform_block_index_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_11)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    if (uniform_condition_2) {
        uniform_reused_word_4 =
            uniform_address_word_6 * uniform_index_word_20 + uniform_index_word_20;
    }

    packed_half_2 = max(int32_t(0u), int32_t(uniform_address_word_14));
    if (!condition_0) {
        matrix_fragment_6 = uniform_block_index_5;

        matrix_fragment_4 = uint32_t(uint32_t(packed_half_2) >> 0x3u);

        matrix_fragment_6 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_10));
        do {
            condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_6);
            if (!condition_0) {
                matrix_fragment_8 = packed_half_0 * uniform_block_index_5;

                matrix_fragment_5 = matrix_fragment_4;
                do {
                    packed_half_2 = parameters[parameter_wait_optional_low_10];
                    packed_half_3 = parameters[parameter_wait_optional_high_11];

                    matrix_fragment_7 = matrix_fragment_8 + matrix_fragment_5;

                    split_address_words(packed_half_2, packed_half_3,
                                        int64_t(int32_t(matrix_fragment_7)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(packed_half_2, packed_half_3));

                    matrix_fragment_7 = pipeline_read<CheckBounds>(
                        combine_words(packed_half_2, packed_half_3), memory_ranges, error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_6);

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
            packed_half_0 = packed_half_0 + 0x1u;

            condition_0 = uint32_t(packed_half_0) != uint32_t(uniform_index_word_9);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    uniform_address_word_14 = parameters[parameter_tensor_height_6];
    uniform_address_word_15 = parameters[parameter_tensor_width_7];

    matrix_fragment_15 = threadIdx.x & 31u;

    matrix_accumulator_138 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_139 = parameters[parameter_trained_weights_high_5];

    uniform_address_word_6 = uint32_t(int32_t(uniform_address_word_7) >> 0x1fu);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_condition_3 = false;

    uniform_condition_2 = true;

    uniform_address_word_6 =
        uint32_t(combine_words(uniform_address_word_6, 0u) << 0x2u >> 32) + uniform_address_word_7;

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x2u >> 32) + uniform_index_word_8;

    uniform_condition_6 = false;

    packed_half_134 = parameters[parameter_trained_weights_low_4];
    packed_half_135 = parameters[parameter_trained_weights_high_5];

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    __syncthreads();

    uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_14) >> 0x1fu);

    uniform_index_word_16 = uniform_address_word_14 & 0xfffffffcu;

    uniform_index_word_13 = uint32_t(int32_t(uniform_address_word_15) >> 0x1fu);

    uniform_index_word_25 = uniform_address_word_15 & 0xfffffffcu;

    uniform_index_word_12 =
        uint32_t(combine_words(uniform_index_word_12, 0u) << 0x2u >> 32) + uniform_address_word_14;

    uniform_condition_1 = uint32_t(uniform_index_word_16) != uint32_t(0x4u);

    uniform_address_word_14 =
        uint32_t(combine_words(uniform_index_word_13, 0u) << 0x2u >> 32) + uniform_address_word_15;

    address_word_162 = parameters[parameter_trained_weights_low_4];
    address_word_163 = parameters[parameter_trained_weights_high_5];

    uniform_index_word_13 = uint32_t(int32_t(uniform_index_word_12) >> 0x2u);

    uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_6) >> 0x2u);

    uniform_index_word_20 = uint32_t(int32_t(uniform_address_word_14) >> 0x2u);

    uniform_index_word_21 = uniform_index_word_12 + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_20);

    split_address_words(matrix_accumulator_64, matrix_accumulator_65,
                        int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_accumulator_138, matrix_accumulator_139));

    if (uniform_condition_1) {
        uniform_address_word_6 = uniform_index_word_11 + 0x1u;

        uniform_condition_4 = int32_t(uniform_index_word_11) >= int32_t(uniform_index_word_13);
    }

    uniform_condition_0 =
        int32_t(uniform_address_word_7) > int32_t(0u - 0x4u) && !uniform_condition_0;

    if (uniform_condition_1) {
        uniform_condition_5 = int32_t(uniform_address_word_6) >= int32_t(uniform_index_word_13);

        uniform_condition_3 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x3u) || uniform_condition_4;
    }

    uniform_index_word_23 = uint32_t(uniform_condition_0);

    uniform_condition_4 = false;

    if (uniform_condition_1) {
        uniform_condition_4 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x7u) || uniform_condition_5;

        uniform_condition_2 = !uniform_condition_3;
    }

    uniform_condition_5 = int32_t(uniform_index_word_23) != int32_t(0u);

    uniform_condition_0 = uint32_t(uniform_index_word_25) == uint32_t(0x4u) || uniform_condition_3;

    uniform_index_word_16 = uniform_condition_3 ? uniform_index_word_12 : 0u;

    uniform_condition_3 = true;

    if (uniform_condition_1) {
        uniform_condition_3 = !uniform_condition_4;
    }

    uniform_condition_2 = uniform_condition_2 & (uniform_condition_0 | uniform_condition_5);

    uniform_condition_0 = uint32_t(uniform_index_word_25) == uint32_t(0x4u) || uniform_condition_4;

    if (uniform_condition_1) {
        uniform_address_word_6 = uniform_index_word_11 + 0x1u;
    }

    uniform_packed_half_22 = uniform_condition_4 ? uniform_index_word_12 : 0u;

    if (uniform_condition_1) {
        uniform_condition_4 = int32_t(uniform_index_word_11) >= int32_t(uniform_index_word_13);
    }

    uniform_condition_3 = uniform_condition_3 & (uniform_condition_0 | uniform_condition_5);

    if (uniform_condition_1) {
        uniform_condition_0 = int32_t(uniform_address_word_6) >= int32_t(uniform_index_word_13);
    }

    uniform_condition_5 = true;

    if (uniform_condition_1) {
        uniform_condition_6 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x3u) || uniform_condition_4;
    }

    uniform_condition_4 = false;

    if (uniform_condition_1) {
        uniform_condition_4 =
            int32_t(uniform_index_word_8) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    uniform_condition_0 = int32_t(uniform_index_word_21) >= int32_t(uniform_index_word_20);

    uniform_index_word_24 = uniform_condition_6 ? 0x40u : 0u;

    if (uniform_condition_2) {
        uniform_address_word_14 = parameters[parameter_input_low_0];
        uniform_address_word_15 = parameters[parameter_input_high_1];
    }

    condition_0 = uniform_condition_2;

    load_four_words_cached<CheckBounds>(combine_words(matrix_accumulator_64, matrix_accumulator_65),
                                        matrix_fragment_36, matrix_fragment_37, matrix_fragment_38,
                                        matrix_fragment_39, memory_ranges, error_counts);

    condition_1 = uniform_condition_3;

    uniform_condition_0 =
        int32_t(uniform_address_word_7) > int32_t(0u - 0x8u) && !uniform_condition_0;

    if (uniform_condition_1) {
        uniform_condition_5 = !uniform_condition_4;
    }

    uniform_condition_6 = uint32_t(uniform_index_word_25) == uint32_t(0x4u) || uniform_condition_4;

    uniform_address_word_6 = 0u;

    if (uniform_condition_1) {
        uniform_address_word_6 = uniform_index_word_11 * uniform_index_word_20;
    }

    if (!uniform_condition_2) {
        uniform_index_word_13 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    uniform_condition_5 = uniform_condition_5 & (uniform_condition_6 | uniform_condition_0);

    uniform_condition_6 = uint32_t(uniform_index_word_25) != uint32_t(0x4u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x200ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    if (!uniform_condition_2) {
        uniform_index_word_13 =
            permute_signed(uniform_index_word_13, uniform_index_word_13, 0x5410u);
    }

    uniform_index_word_16 = !uniform_condition_6 ? uniform_index_word_16 : uniform_index_word_12;

    uniform_packed_half_22 = !uniform_condition_6 ? uniform_packed_half_22 : uniform_index_word_12;

    packed_half_0 = matrix_fragment_15;

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1000ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    if (uniform_condition_2) {
        uniform_address_word_6 = uniform_index_word_16 + uniform_address_word_6;
    }

    if (!condition_0) {
        matrix_fragment_8 = uniform_index_word_13;

        matrix_fragment_9 = uniform_index_word_13;

        matrix_fragment_10 = uniform_index_word_13;
    }

    if (uniform_condition_2) {
        uniform_address_word_6 = uniform_address_word_6 << (0x7u & 31u);
    }

    if (!condition_0) {
        packed_half_11 = uniform_index_word_13;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1200ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    if (uniform_condition_3) {
        uniform_index_word_16 = parameters[parameter_input_low_0];
        uniform_parameter_word_17 = parameters[parameter_input_high_1];
    }

    packed_half_134 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_134, packed_half_135) + 0x4c60ull, memory_ranges, error_counts);

    if (uniform_condition_2) {
        split_address_words(uniform_address_word_14, uniform_address_word_15,
                            int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_address_word_14, uniform_address_word_15));
    }

    uniform_address_word_6 = 0u;

    if (uniform_condition_1) {
        uniform_address_word_6 =
            uniform_index_word_11 * uniform_index_word_20 + uniform_index_word_20;
    }

    split_address_words(packed_half_2, packed_half_3,
                        combine_words(uniform_address_word_14, uniform_address_word_15));

    if (uniform_condition_3) {
        uniform_address_word_6 = uniform_packed_half_22 + uniform_address_word_6;
    }

    if (!uniform_condition_3) {
        uniform_packed_half_22 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
    }

    if (condition_0) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    if (uniform_condition_3) {
        uniform_address_word_6 = uniform_address_word_6 << (0x7u & 31u);
    }

    if (!uniform_condition_3) {
        uniform_packed_half_22 =
            permute_signed(uniform_packed_half_22, uniform_packed_half_22, 0x5410u);
    }

    packed_half_0 = packed_half_0 & 0xfffffffcu;

    if (uniform_condition_5) {
        uniform_address_word_14 = parameters[parameter_input_low_0];
        uniform_address_word_15 = parameters[parameter_input_high_1];
    }

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(packed_half_2, packed_half_3), matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, packed_half_11, memory_ranges, error_counts);
    }

    if (uniform_condition_3) {
        split_address_words(uniform_index_word_16, uniform_parameter_word_17,
                            int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_index_word_16, uniform_parameter_word_17));
    }

    if (!condition_1) {
        matrix_fragment_16 = uniform_packed_half_22;

        matrix_fragment_17 = uniform_packed_half_22;

        matrix_fragment_18 = uniform_packed_half_22;

        matrix_fragment_19 = uniform_packed_half_22;
    }

    packed_half_151 = matrix_fragment_15 + 0x4u + (0u - packed_half_0);

    uniform_condition_3 = int32_t(uniform_index_word_24) != int32_t(0u);

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        combine_words(uniform_index_word_16, uniform_parameter_word_17));

    if (condition_1) {
        split_address_words(matrix_fragment_4, matrix_fragment_5,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_4, matrix_fragment_5));

        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_4, matrix_fragment_5), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    packed_half_3 = 0u - packed_half_0 + matrix_fragment_15;

    split_address_words(matrix_fragment_6, matrix_fragment_7,
                        int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_138, matrix_accumulator_139));

    matrix_fragment_48 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_6, matrix_fragment_7) + 0x2010ull, memory_ranges,
        error_counts);

    split_address_words(packed_half_150, packed_half_151,
                        uint64_t(uint32_t(packed_half_151)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_138, matrix_accumulator_139));

    packed_half_161 = matrix_fragment_15 + 0x8u + (0u - packed_half_0);

    matrix_fragment_56 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_150, packed_half_151) + 0x2010ull, memory_ranges, error_counts);

    split_address_words(packed_half_160, packed_half_161,
                        uint64_t(uint32_t(packed_half_161)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_138, matrix_accumulator_139));

    packed_half_3 = matrix_fragment_15 + 0xcu + (0u - packed_half_0);

    matrix_fragment_55 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, packed_half_161) + 0x2010ull, memory_ranges, error_counts);

    split_address_words(matrix_accumulator_138, matrix_accumulator_139,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_138, matrix_accumulator_139));

    matrix_fragment_54 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_138, matrix_accumulator_139) + 0x2010ull, memory_ranges,
        error_counts);

    uniform_index_word_13 = uniform_condition_3 ? uniform_index_word_21 : 0u;

    uniform_address_word_6 = 0u;

    if (uniform_condition_1) {
        uniform_address_word_6 =
            uniform_index_word_11 * uniform_index_word_20 + uniform_index_word_20;
    }

    uniform_index_word_13 = !uniform_condition_6 ? uniform_index_word_13 : uniform_index_word_21;

    if (!uniform_condition_6) {
        uniform_index_word_21 = uniform_condition_4 ? uniform_index_word_21 : 0u;
    }

    if (uniform_condition_5) {
        uniform_address_word_6 = uniform_index_word_21 + uniform_address_word_6;
    }

    condition_0 = uniform_condition_5;

    if (uniform_condition_5) {
        uniform_address_word_6 = uniform_address_word_6 << (0x7u & 31u);
    }

    if (!uniform_condition_5) {
        uniform_index_word_16 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_5) {
        split_address_words(uniform_address_word_14, uniform_address_word_15,
                            int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_address_word_14, uniform_address_word_15));
    }

    if (!uniform_condition_5) {
        uniform_index_word_16 =
            permute_signed(uniform_index_word_16, uniform_index_word_16, 0x5410u);
    }

    thread_scratch[0] = matrix_fragment_6;
    thread_scratch[1] = matrix_fragment_7;

    split_address_words(packed_half_2, packed_half_3,
                        combine_words(uniform_address_word_14, uniform_address_word_15));

    if (!condition_0) {
        matrix_fragment_4 = uniform_index_word_16;

        matrix_fragment_5 = uniform_index_word_16;
    }

    uniform_condition_2 = true;

    if (uniform_condition_1) {
        uniform_condition_2 = !uniform_condition_3;
    }

    if (condition_0) {
        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));
    }

    if (!condition_0) {
        matrix_fragment_6 = uniform_index_word_16;

        matrix_fragment_7 = uniform_index_word_16;
    }

    uniform_condition_3 = uint32_t(uniform_index_word_25) == uint32_t(0x4u) || uniform_condition_3;

    uniform_address_word_6 = 0u;

    packed_half_0 = 0x2b28u;

    uniform_index_word_16 = parameters[parameter_tensor_height_6];
    uniform_parameter_word_17 = parameters[parameter_tensor_width_7];

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(packed_half_2, packed_half_3), matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    uniform_condition_2 = uniform_condition_2 & (uniform_condition_3 | uniform_condition_0);

    if (uniform_condition_1) {
        uniform_address_word_6 = uniform_index_word_11 * uniform_index_word_20;
    }

    if (uniform_condition_2) {
        uniform_address_word_14 = parameters[parameter_input_low_0];
        uniform_address_word_15 = parameters[parameter_input_high_1];

        uniform_address_word_6 = uniform_index_word_13 + uniform_address_word_6;
    }

    condition_0 = uniform_condition_2;

    uniform_index_word_13 = parameters[parameter_geometry_or_control_9];

    if (uniform_condition_2) {
        uniform_address_word_6 = uniform_address_word_6 << (0x7u & 31u);

        split_address_words(uniform_address_word_14, uniform_address_word_15,
                            int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_address_word_14, uniform_address_word_15));
    }

    if (!uniform_condition_2) {
        uniform_address_word_6 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));

        uniform_address_word_6 =
            permute_signed(uniform_address_word_6, uniform_address_word_6, 0x5410u);
    }

    split_address_words(matrix_fragment_12, packed_half_13,
                        combine_words(uniform_address_word_14, uniform_address_word_15));

    if (!condition_0) {
        matrix_fragment_20 = uniform_address_word_6;

        matrix_fragment_21 = uniform_address_word_6;

        matrix_fragment_22 = uniform_address_word_6;

        matrix_fragment_23 = uniform_address_word_6;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_12, packed_half_13,
                            int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_12, packed_half_13));
    }

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, packed_half_13), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_16, matrix_fragment_17,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_36,
                             matrix_fragment_37, 0u, 0u);

    uniform_index_word_20 = uint32_t(int32_t(uniform_index_word_16) >> 0x1fu);

    uniform_index_word_21 = uint32_t(int32_t(uniform_parameter_word_17) >> 0x1fu);

    uniform_condition_6 = int32_t(uniform_index_word_23) != int32_t(0u);

    uniform_address_word_14 = parameters[parameter_output_low_2];
    uniform_address_word_15 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

    packed_half_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x44004400u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x44004400u);

    packed_half_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0xc400c400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_13),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_13),
        select_half_operand<2, false, false>(matrix_fragment_12), 0x3b283b28u);

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(packed_half_13), 0x3b283b28u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

    packed_half_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0xc400c400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_13),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_13),
        select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    packed_half_13 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                         select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x44004400u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_12,
                             packed_half_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    packed_half_13 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                         select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_14),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_12,
                             packed_half_13);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_10);

    packed_half_13 = unpack_fp8_bits(packed_half_11);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_accumulator_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_accumulator_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    packed_half_13 = unpack_fp8_bits(packed_half_11 >> 16);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_3 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                         select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0xc400c400u);

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_accumulator_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_72,
                             matrix_accumulator_73);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_3),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_3),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_12,
                             packed_half_13);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x400ull, matrix_fragment_12,
        packed_half_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_61));

    packed_half_2 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(packed_half_3));

    packed_half_2 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_accumulator_60));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_46, matrix_fragment_47, packed_half_2, packed_half_3);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_18);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_19);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_55));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_16);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_accumulator_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x600ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, packed_half_2, packed_half_3);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_54));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_60,
                             matrix_accumulator_61);

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_40, matrix_fragment_41,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_30,
                             matrix_fragment_31, packed_half_2, packed_half_3);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0xc400c400u);

    matrix_accumulator_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0xc400c400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3b283b28u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_fragment_69), 0x3b283b28u);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_63));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_76),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3b283b28u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_32),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_68),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3b283b28u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_4);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_76 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(matrix_accumulator_76));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_fragment_41 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_40,
                             matrix_fragment_41);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1400ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_accumulator_92 = unpack_fp8_bits(matrix_fragment_20);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_32),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_93 = unpack_fp8_bits(matrix_fragment_21);

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_34),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<2, false, false>(matrix_fragment_35), 0x3b283b28u);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_accumulator_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0xc400c400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_48), 0x3b283b28u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0xc400c400u);

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_32),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0xc400c400u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_63),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<2, false, false>(matrix_fragment_34), 0x3b283b28u);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<2, false, false>(matrix_fragment_84), 0x3b283b28u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_85),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(matrix_fragment_48));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1600ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_57),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x800ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<2, false, false>(matrix_fragment_86), 0x3b283b28u);

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<2, false, false>(packed_half_62), 0x3b283b28u);

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_62));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_92,
                             matrix_accumulator_93);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_56));

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_22);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_44,
                             matrix_fragment_45);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0xa00ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    multiply_accumulate_half(matrix_accumulator_90, packed_half_91, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

    matrix_fragment_57 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

    packed_half_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_48 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

    matrix_fragment_49 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x44004400u);

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_91), 0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3b283b28u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_98),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_98),
        select_half_operand<2, false, false>(packed_half_99), 0x3b283b28u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_102),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0xc400c400u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0xc400c400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_108),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_12, packed_half_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                            select_half_operand<2, false, false>(packed_half_105));

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<2, false, false>(packed_half_109), 0x3b283b28u);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0xc400c400u);

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    packed_half_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_109));

    matrix_accumulator_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    packed_half_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(packed_half_100));

    matrix_accumulator_96 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    packed_half_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_97 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(packed_half_106));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_106 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_51 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0xc400c400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(packed_half_99));

    packed_half_99 = maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                        0xc400c400u);

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_96),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(packed_half_98));

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96),
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_accumulator_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_99),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_99),
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3b283b28u);

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x44004400u);

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    packed_half_98 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                         select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_30),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x44004400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    packed_half_99 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                         select_half_operand<2, false, false>(packed_half_99));

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    matrix_fragment_30 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_fragment_29 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_29),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_31),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_29),
        select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_31),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_12),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_51));

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<2, false, false>(packed_half_13), 0x3b283b28u);

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_28),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_accumulator_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0xc400c400u);

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<2, false, false>(matrix_fragment_29), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0xc400c400u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_116),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                       select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_12),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_91),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                          select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                       select_half_operand<2, false, false>(packed_half_104));

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116),
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3b283b28u);

    packed_half_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<2, false, false>(packed_half_13), 0x3b283b28u);

    packed_half_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_30),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_12 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_115),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                    select_half_operand<2, false, false>(0u));

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0xc400c400u);

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_52,
                             matrix_fragment_53);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<2, false, false>(matrix_fragment_31), 0x3b283b28u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_115),
        select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1a00ull, matrix_fragment_12,
        packed_half_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_114));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1800ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                       select_half_operand<2, false, false>(packed_half_105));

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_25),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_55 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                      select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                          select_half_operand<2, false, false>(packed_half_115));

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3b283b28u);

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                                         select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                          select_half_operand<2, false, false>(packed_half_91));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0xe00ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_112));

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(0u));

    packed_half_115 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                       select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(packed_half_115));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_100));

    packed_half_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                    select_half_operand<2, false, false>(0u));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_90, packed_half_91, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_60,
                             matrix_accumulator_61);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_98),
                                          select_half_operand<2, false, false>(0u));

    packed_half_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_99),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(packed_half_110));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(packed_half_98));

    packed_half_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                   select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_63 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                      select_half_operand<2, false, false>(matrix_accumulator_102));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_accumulator_60,
                             matrix_accumulator_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_40, matrix_fragment_41, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_78,
                             matrix_accumulator_79);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_60,
                             matrix_accumulator_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_42, matrix_fragment_43, matrix_accumulator_80,
                             matrix_accumulator_81);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0xc00ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_86,
                             matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_38, matrix_fragment_39, packed_half_2, packed_half_3);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_accumulator_60,
                             matrix_accumulator_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_accumulator_60,
                             matrix_accumulator_61, packed_half_62, matrix_accumulator_63,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_68,
                             matrix_fragment_69);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    packed_half_100 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_accumulator_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_100), 0xc400c400u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0xc400c400u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(packed_half_98), 0x3b283b28u);

    packed_half_104 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0xc400c400u);

    matrix_accumulator_88 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_88 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0xc400c400u);

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(packed_half_105), 0x3b283b28u);

    packed_half_106 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x44004400u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_88),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, packed_half_11, matrix_fragment_44,
                             matrix_fragment_45, 0u, 0u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                          select_half_operand<2, false, false>(packed_half_98));

    packed_half_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_99 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                         select_half_operand<2, false, false>(packed_half_99));

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    matrix_accumulator_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0x44004400u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_accumulator_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    packed_half_98 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_109));

    packed_half_109 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_88), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_57),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(packed_half_112));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_98), 0xc400c400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

    packed_half_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_106));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_50 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                                             select_half_operand<2, false, false>(packed_half_101));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                         select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0x44004400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    matrix_accumulator_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_accumulator_89));

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

    packed_half_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(packed_half_107));

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                          select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    packed_half_98 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                         select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_38),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_44),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_38),
        select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    matrix_fragment_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_46),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_44),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_47),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_111));

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                  select_half_operand<2, false, false>(0u));

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                          select_half_operand<2, false, false>(packed_half_110));

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<2, false, false>(matrix_fragment_46), 0x3b283b28u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(0u));

    packed_half_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(packed_half_105));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_118),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(packed_half_3));

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_118),
        select_half_operand<2, false, false>(packed_half_119), 0x3b283b28u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_45 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_99),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0xc400c400u);

    matrix_fragment_36 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x44004400u);

    matrix_fragment_37 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x44004400u);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

    packed_half_119 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(packed_half_119));

    matrix_accumulator_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_2),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1c00ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    matrix_accumulator_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x1e00ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_2),
        select_half_operand<2, false, false>(packed_half_3), 0x3b283b28u);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_116),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116),
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3b283b28u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_2 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_2), 0xc400c400u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_89),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_117 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_2),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_accumulator_88), 0x3b283b28u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_89),
        select_half_operand<2, false, false>(packed_half_99), 0x3b283b28u);

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                          select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_2),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    matrix_accumulator_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(matrix_accumulator_88));

    packed_half_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(packed_half_99));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_115));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(packed_half_3));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_99),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                      select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_accumulator_60));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_90,
                             packed_half_91);

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_52, matrix_fragment_53,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_30,
                             matrix_fragment_31, matrix_accumulator_78, matrix_accumulator_79);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_82,
                             matrix_accumulator_83);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_12, packed_half_13, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_68,
                             matrix_fragment_69);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2260ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_12, packed_half_13, matrix_accumulator_72,
                             matrix_accumulator_73);

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_70,
                             matrix_fragment_71);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_12, packed_half_13, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_96,
                             matrix_accumulator_97);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2060ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    multiply_accumulate_half(packed_half_114, packed_half_115, matrix_fragment_8, matrix_fragment_9,
                             matrix_fragment_10, packed_half_11, matrix_fragment_24,
                             matrix_fragment_25, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_12, packed_half_13, matrix_fragment_84,
                             matrix_fragment_85);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_86,
                             matrix_fragment_87);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2460ull, matrix_fragment_12,
        packed_half_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_114), 0x44004400u);

    matrix_fragment_10 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x44004400u);

    matrix_fragment_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

    packed_half_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0xc400c400u);

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_8),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    matrix_fragment_9 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x44004400u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_10),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_10),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_fragment_8 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x44004400u);

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0xc400c400u);

    matrix_fragment_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_11),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_11),
        select_half_operand<2, false, false>(matrix_fragment_20), 0x3b283b28u);

    packed_half_11 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                                        0x44004400u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_9),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_11 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_11), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_21 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x44004400u);

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_9),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_10),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_11),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_10),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_11),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2660ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, packed_half_11, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_91),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_20 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_91),
        select_half_operand<2, false, false>(packed_half_98), 0x3b283b28u);

    packed_half_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x44004400u);

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

    packed_half_99 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                                        0x44004400u);

    packed_half_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_91), 0xc400c400u);

    packed_half_104 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x44004400u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_20),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_99), 0xc400c400u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_91),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_106 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x44004400u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0xc400c400u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    packed_half_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0xc400c400u);

    packed_half_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_91),
        select_half_operand<2, false, false>(packed_half_99), 0x3b283b28u);

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(matrix_accumulator_82));

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_105),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    packed_half_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_104),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                            select_half_operand<2, false, false>(packed_half_98));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_accumulator_73 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                            select_half_operand<2, false, false>(packed_half_99));

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<2, false, false>(packed_half_106), 0x3b283b28u);

    matrix_fragment_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    packed_half_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<2, false, false>(packed_half_91), 0x3b283b28u);

    matrix_fragment_17 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    packed_half_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                            select_half_operand<2, false, false>(matrix_accumulator_74));

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_107),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0xc400c400u);

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(packed_half_106));

    matrix_accumulator_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(packed_half_91));

    matrix_accumulator_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0xc400c400u);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    matrix_accumulator_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3b283b28u);

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_74),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_63),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3b283b28u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_accumulator_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    matrix_accumulator_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_40),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_fragment_4 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    matrix_accumulator_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_4),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_7),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0xc400c400u);

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_4),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_7),
        select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_43),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0xc400c400u);

    matrix_fragment_4 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_4), 0x3b283b28u);

    matrix_accumulator_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    matrix_accumulator_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_62),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_91 = maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                                        0xc400c400u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0xc400c400u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_accumulator_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_60),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                    select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_91),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_63),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                      select_half_operand<2, false, false>(0u));

    packed_half_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_91),
        select_half_operand<2, false, false>(packed_half_98), 0x3b283b28u);

    packed_half_99 = maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                                        0xc400c400u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3b283b28u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<2, false, false>(packed_half_62), 0x3b283b28u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2860ull,
        matrix_accumulator_60, matrix_accumulator_61, packed_half_62, matrix_accumulator_63,
        memory_ranges, error_counts);

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_accumulator_64, matrix_accumulator_65) + 0x2a60ull,
        matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67,
        memory_ranges, error_counts);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0xc400c400u);

    packed_half_98 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                         select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                    select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_5 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_7),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_115 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                    select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_7),
        select_half_operand<2, false, false>(matrix_fragment_5), 0x3b283b28u);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_99),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_91 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                         select_half_operand<2, false, false>(packed_half_91));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_20));

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_99),
        select_half_operand<2, false, false>(packed_half_104), 0x3b283b28u);

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                          select_half_operand<2, false, false>(packed_half_104));

    packed_half_114 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                    select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0xc400c400u);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_20),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_22),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_19),
        select_half_operand<0, false, true>(packed_half_0), 0x37283728u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<2, false, false>(matrix_fragment_21), 0x3b283b28u);

    matrix_fragment_23 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_22),
        select_half_operand<2, false, false>(matrix_fragment_23), 0x3b283b28u);

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_110));

    packed_half_0 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<2, false, false>(packed_half_0), 0x3b283b28u);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(0u));

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_91),
                                       select_half_operand<2, false, false>(packed_half_104));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, packed_half_112,
                             packed_half_113, packed_half_114, packed_half_115, matrix_fragment_44,
                             matrix_fragment_45, matrix_accumulator_94, matrix_accumulator_95);

    multiply_accumulate_half(packed_half_100, packed_half_101, packed_half_112, packed_half_113,
                             packed_half_114, packed_half_115, matrix_fragment_46,
                             matrix_fragment_47, packed_half_100, packed_half_101);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, packed_half_112,
                             packed_half_113, packed_half_114, packed_half_115, matrix_fragment_36,
                             matrix_fragment_37, matrix_accumulator_116, matrix_accumulator_117);

    multiply_accumulate_half(packed_half_114, packed_half_115, packed_half_112, packed_half_113,
                             packed_half_114, packed_half_115, matrix_fragment_38,
                             matrix_fragment_39, matrix_fragment_70, matrix_fragment_71);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_46, matrix_fragment_47, packed_half_2, packed_half_3);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(packed_half_118, packed_half_119, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_38,
                             matrix_fragment_39, matrix_fragment_86, matrix_fragment_87);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_68,
                             matrix_fragment_69);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_76,
                             matrix_accumulator_77);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_46, matrix_fragment_47, matrix_accumulator_102,
                             matrix_accumulator_103);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_84,
                             matrix_fragment_85);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                       select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_118),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                                      select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                       select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                  select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_56,
                             matrix_fragment_57, 0u, 0u);

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_71 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    multiply_accumulate_half(matrix_accumulator_90, packed_half_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                      select_half_operand<2, false, false>(packed_half_104),
                                      select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                        select_half_operand<2, false, false>(packed_half_128));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                      select_half_operand<2, false, false>(packed_half_105),
                                      select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_136, packed_half_137, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_12, packed_half_13, 0u, 0u);

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                         select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(matrix_fragment_26),
                                      select_half_operand<2, false, false>(packed_half_13));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                             select_half_operand<2, false, false>(packed_half_129));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(packed_half_13));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    packed_half_0 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(matrix_fragment_27),
                                      select_half_operand<2, false, false>(packed_half_0));

    packed_half_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(packed_half_13));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(packed_half_0));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_2 = result_word;
    }

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    packed_half_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                    select_half_operand<2, false, false>(packed_half_3));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_0), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_3 = result_word;
    }

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                                      select_half_operand<2, false, false>(packed_half_91),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_accumulator_131),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_13), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_32 = result_word;
    }

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_33 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                   select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(packed_half_3));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_2), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_3 = result_word;
    }

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_10, packed_half_11, 0u, 0u);

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                        select_half_operand<2, false, false>(matrix_fragment_32));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_13 = result_word;
    }

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_34), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_35), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_0 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, packed_half_11, 0u, 0u);

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(packed_half_3));

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_accumulator_80));

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, packed_half_11, 0u, 0u);

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                                      select_half_operand<2, false, false>(matrix_accumulator_78),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                    select_half_operand<2, false, false>(packed_half_13));

    multiply_accumulate_half(packed_half_142, packed_half_143, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_9 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                            select_half_operand<2, false, false>(packed_half_106));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                                      select_half_operand<2, false, false>(matrix_accumulator_82),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    multiply_accumulate_half(packed_half_144, packed_half_145, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_10, packed_half_11, 0u, 0u);

    matrix_fragment_84 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_fragment_14 = permute_signed(packed_half_13, packed_half_13, 0x5432u);

    matrix_fragment_10 = permute_signed(packed_half_3, packed_half_3, 0x5432u);

    packed_half_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                                      select_half_operand<2, false, false>(matrix_accumulator_72),
                                      select_half_operand<2, false, false>(packed_half_11));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                   select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(matrix_accumulator_146));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_accumulator_81));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                        select_half_operand<2, false, false>(packed_half_107));

    packed_half_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                    select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10), 0x4100410u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<2, false, false>(matrix_accumulator_77));

    matrix_fragment_34 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_35 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_35));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                                      select_half_operand<2, false, false>(matrix_accumulator_79),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(matrix_accumulator_83),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_14 & 65535u) * 65537u);

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                                      select_half_operand<2, false, false>(matrix_accumulator_73),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_34 = half_to_float_bits((matrix_fragment_14 >> 16) * 65537u);

    packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                    select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_11), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_110 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_97 = result_word;
    }

    matrix_accumulator_95 = rsqrt_bits(matrix_fragment_34);

    packed_half_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0x4100410u);

    matrix_fragment_84 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                        select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(matrix_accumulator_147));

    matrix_accumulator_94 = half_to_float_bits((packed_half_13 & 65535u) * 65537u);

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                          select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_34 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                             select_half_operand<2, false, false>(packed_half_144));

    packed_half_99 = half_to_float_bits((packed_half_13 >> 16) * 65537u);

    packed_half_111 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_85));

    packed_half_13 = rsqrt_bits(matrix_accumulator_94);

    matrix_accumulator_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_fragment_10 = pack_half(matrix_accumulator_95, matrix_fragment_10);

    matrix_accumulator_95 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_accumulator_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                        select_half_operand<2, false, false>(packed_half_111));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_60, matrix_accumulator_61, 0u, 0u);

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                       select_half_operand<2, false, false>(matrix_accumulator_94));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_108 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_84), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_85 = result_word;
    }

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(packed_half_143));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_35), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_98 = result_word;
    }

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_145));

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_20, matrix_fragment_21,
                             matrix_fragment_22, matrix_fragment_23, packed_half_62,
                             matrix_accumulator_63, 0u, 0u);

    packed_half_112 = rsqrt_bits(packed_half_99);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    matrix_fragment_22 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                        select_half_operand<2, false, false>(packed_half_110));

    packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                    select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_fragment_12 = threadIdx.x & 31u;

    matrix_fragment_21 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                      select_half_operand<2, false, false>(packed_half_137),
                                      select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_20 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(matrix_accumulator_141),
                                      select_half_operand<2, false, false>(packed_half_111));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_96), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_109 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_34), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_111 = result_word;
    }

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_22), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_99 = result_word;
    }

    matrix_fragment_23 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                        select_half_operand<2, false, false>(packed_half_108));

    matrix_accumulator_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                       select_half_operand<2, false, false>(matrix_fragment_85));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_108 = result_word;
    }

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = permute_signed(matrix_accumulator_97, matrix_accumulator_97, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_23), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_98 = result_word;
    }

    packed_half_13 = pack_half(packed_half_112, packed_half_13);

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                       select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_11), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_20 = result_word;
    }

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(packed_half_111));

    split_address_words(address_word_162, address_word_163,
                        int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_162, address_word_163));

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(packed_half_13));

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                          select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_13));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_97 = result_word;
    }

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                         select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_110 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_34), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_109 = result_word;
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x2c60ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_fragment_22 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(packed_half_99));

    packed_half_108 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                                     select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_21 = permute_signed(matrix_fragment_22, matrix_fragment_22, 0x5432u);

    matrix_fragment_23 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                        select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_22 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                    select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_20 = permute_signed(matrix_fragment_23, matrix_fragment_23, 0x5432u);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x4100410u);

    packed_half_111 = half_to_float_bits((matrix_fragment_87 & 65535u) * 65537u);

    packed_half_148 = half_to_float_bits((matrix_fragment_87 >> 16) * 65537u);

    matrix_fragment_87 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                       select_half_operand<2, false, false>(matrix_accumulator_97));

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_60,
                             matrix_accumulator_61, 0u, 0u);

    packed_half_110 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                     select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_34 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(packed_half_109));

    packed_half_109 = half_to_float_bits((matrix_fragment_22 & 65535u) * 65537u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_62, matrix_accumulator_63, 0u, 0u);

    multiply_accumulate_half(packed_half_98, packed_half_99, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_64,
                             matrix_accumulator_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    matrix_fragment_7 = half_to_float_bits((matrix_fragment_22 >> 16) * 65537u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x2e60ull, matrix_fragment_20,
        matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_108), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_135 = result_word;
    }

    matrix_fragment_6 = rsqrt_bits(packed_half_109);

    matrix_fragment_7 = rsqrt_bits(matrix_fragment_7);

    matrix_fragment_5 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x4100410u);

    packed_half_111 = rsqrt_bits(packed_half_111);

    packed_half_148 = rsqrt_bits(packed_half_148);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(packed_half_135));

    packed_half_108 = permute_signed(packed_half_11, packed_half_11, 0x5432u);

    matrix_fragment_6 = pack_half(matrix_fragment_7, matrix_fragment_6);

    packed_half_135 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                     select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_7 = permute_signed(packed_half_110, packed_half_110, 0x5432u);

    matrix_fragment_4 = half_to_float_bits((matrix_fragment_5 & 65535u) * 65537u);

    packed_half_135 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_135), 0x4100410u);

    matrix_fragment_5 = half_to_float_bits((matrix_fragment_5 >> 16) * 65537u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_149 = half_to_float_bits((packed_half_135 >> 16) * 65537u);

    packed_half_11 = pack_half(packed_half_148, packed_half_111);

    packed_half_148 = half_to_float_bits((packed_half_135 & 65535u) * 65537u);

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    packed_half_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3060ull, packed_half_108,
        packed_half_109, packed_half_110, packed_half_111, memory_ranges, error_counts);

    matrix_fragment_7 = permute_signed(matrix_fragment_34, matrix_fragment_34, 0x5432u);

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_135 = half_to_float_bits((packed_half_152 & 65535u) * 65537u);

    packed_half_148 = rsqrt_bits(packed_half_148);

    packed_half_154 = half_to_float_bits((packed_half_152 >> 16) * 65537u);

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x4100410u);

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(packed_half_11));

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(packed_half_11));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(packed_half_11));

    packed_half_135 = rsqrt_bits(packed_half_135);

    packed_half_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_106),
                                         select_half_operand<2, false, false>(packed_half_11));

    packed_half_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_81 = half_to_float_bits((matrix_accumulator_79 & 65535u) * 65537u);

    packed_half_154 = rsqrt_bits(packed_half_154);

    matrix_accumulator_79 = half_to_float_bits((matrix_accumulator_79 >> 16) * 65537u);

    matrix_accumulator_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_155 = pack_half(matrix_fragment_5, matrix_fragment_4);

    packed_half_107 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_107),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_164 = pack_half(packed_half_149, packed_half_148);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3260ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    matrix_accumulator_81 = rsqrt_bits(matrix_accumulator_81);

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                          select_half_operand<2, false, false>(packed_half_164));

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_79 = rsqrt_bits(matrix_accumulator_79);

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_11),
                                   select_half_operand<2, false, false>(0u));

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                          select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(matrix_fragment_37));

    packed_half_135 = pack_half(packed_half_154, packed_half_135);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<2, false, false>(packed_half_11));

    packed_half_11 = pack_half(0u, packed_half_134);

    packed_half_158 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                            select_half_operand<2, false, false>(packed_half_155));

    packed_half_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(packed_half_155));

    packed_half_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                            select_half_operand<2, false, false>(packed_half_164));

    packed_half_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(packed_half_164));

    packed_half_159 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                          select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_30 = pack_half(matrix_accumulator_79, matrix_accumulator_81);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                       select_half_operand<2, false, false>(0u));

    packed_half_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(packed_half_155));

    matrix_accumulator_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                      select_half_operand<2, false, false>(0u));

    packed_half_164 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<2, false, false>(packed_half_164));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3460ull, matrix_accumulator_72,
        matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                    select_half_operand<2, false, false>(0u));

    packed_half_105 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_105),
                                          select_half_operand<0, false, false>(packed_half_11));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3660ull, matrix_accumulator_76,
        matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3860ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    packed_half_104 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_104),
                                          select_half_operand<0, false, false>(packed_half_11));

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(packed_half_107));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                       select_half_operand<2, false, false>(matrix_fragment_85));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(packed_half_105));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3a60ull, packed_half_104,
        packed_half_105, packed_half_106, packed_half_107, memory_ranges, error_counts);

    matrix_fragment_32 = permute_signed(matrix_fragment_87, matrix_fragment_87, 0x5432u);

    packed_half_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_13),
                                         select_half_operand<0, false, false>(packed_half_11));

    packed_half_134 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                     select_half_operand<2, false, false>(matrix_fragment_32));

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_13),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_134 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_134), 0x4100410u);

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_32 = half_to_float_bits((packed_half_134 & 65535u) * 65537u);

    packed_half_134 = half_to_float_bits((packed_half_134 >> 16) * 65537u);

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_32 = rsqrt_bits(matrix_fragment_32);

    packed_half_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_10 = rsqrt_bits(packed_half_134);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_135));

    packed_half_165 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                          select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_164),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_10 = pack_half(matrix_fragment_10, matrix_fragment_32);

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_134, packed_half_135, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_26 = permute_signed(matrix_fragment_9, matrix_fragment_9, 0x5432u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x4100410u);

    matrix_fragment_9 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5432u);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_27 & 65535u) * 65537u);

    packed_half_165 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_165),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(packed_half_165));

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_148, packed_half_149, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x4100410u);

    matrix_fragment_9 = half_to_float_bits((matrix_fragment_27 >> 16) * 65537u);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_13),
                                       select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_23 = half_to_float_bits((matrix_fragment_22 & 65535u) * 65537u);

    matrix_fragment_9 = rsqrt_bits(matrix_fragment_9);

    packed_half_13 = rsqrt_bits(matrix_fragment_23);

    matrix_fragment_23 = permute_signed(packed_half_0, packed_half_0, 0x5432u);

    matrix_fragment_25 = half_to_float_bits((matrix_fragment_22 >> 16) * 65537u);

    packed_half_0 = add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                   select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_8 = pack_half(matrix_fragment_9, matrix_fragment_8);

    matrix_fragment_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x4100410u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_140),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_142 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = half_to_float_bits((matrix_fragment_22 & 65535u) * 65537u);

    matrix_accumulator_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_22 = half_to_float_bits((matrix_fragment_22 >> 16) * 65537u);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = rsqrt_bits(matrix_fragment_22);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_140 = thread_scratch[0];
    matrix_accumulator_141 = thread_scratch[1];

    packed_half_0 = rsqrt_bits(matrix_fragment_25);

    matrix_fragment_10 = rsqrt_bits(matrix_fragment_10);

    packed_half_159 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_159),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_158),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(packed_half_159));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_20,
                             matrix_fragment_21);

    packed_half_13 = pack_half(packed_half_0, packed_half_13);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(0u));

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_10 = pack_half(matrix_fragment_23, matrix_fragment_10);

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_28, matrix_fragment_29, packed_half_108,
                             packed_half_109);

    packed_half_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                                         select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_137 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_26, matrix_fragment_27, packed_half_110,
                             packed_half_111);

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                       select_half_operand<2, false, false>(packed_half_0));

    packed_half_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                                         select_half_operand<0, false, false>(packed_half_11));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_137),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(packed_half_13));

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5);

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                      select_half_operand<2, false, false>(0u));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_91),
                                      select_half_operand<2, false, false>(packed_half_0));

    packed_half_0 = 0x29c0u;

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_6,
                             matrix_fragment_7);

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                       select_half_operand<2, false, false>(0u));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<0, false, false>(packed_half_11));

    packed_half_11 = permute_signed(packed_half_11, packed_half_0, 0x5410u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                      select_half_operand<2, false, false>(0u));

    packed_half_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_134),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_146 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_146),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_20),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_148 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_148),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_0 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_134), 0x3c203c20u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_146 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_146), 0x3c203c20u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_20 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_72,
                             matrix_accumulator_73);

    packed_half_148 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_148), 0x3c203c20u);

    packed_half_0 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_0), 0x3e473e47u);

    matrix_accumulator_146 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_146), 0x3e473e47u);

    matrix_fragment_20 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x3e473e47u);

    packed_half_148 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_148), 0x3e473e47u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_90, packed_half_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_74,
                             matrix_accumulator_75);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0x3c203c20u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0x3e473e47u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0x3e473e47u);

    packed_half_0 = (packed_half_0 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_76,
                             matrix_accumulator_77);

    packed_half_143 = (packed_half_108 << 0x5u) + 0x7ff88000u;

    packed_half_142 = (packed_half_110 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 = (matrix_accumulator_146 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(packed_half_128, packed_half_129, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_30,
                             matrix_fragment_31, matrix_accumulator_78, matrix_accumulator_79);

    matrix_fragment_10 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_78 = (matrix_fragment_20 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_accumulator_80 = (packed_half_148 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_82,
                             matrix_accumulator_83);

    packed_half_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                                    select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                    select_half_operand<2, false, false>(packed_half_142));

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_24,
                             matrix_fragment_25, packed_half_104, packed_half_105);

    matrix_accumulator_147 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_147),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(packed_half_13));

    packed_half_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_22,
                             matrix_fragment_23, packed_half_106, packed_half_107);

    matrix_fragment_84 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_84),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_147 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_147), 0x3c203c20u);

    packed_half_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_135),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_21),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_149 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_149),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0x3c203c20u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3c203c20u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_128 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_128),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_91),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_129 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_129),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_135), 0x3c203c20u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_147 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_147), 0x3e473e47u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_21 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x3c203c20u);

    packed_half_149 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_149), 0x3c203c20u);

    packed_half_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_13), 0x3e473e47u);

    matrix_fragment_84 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x3e473e47u);

    matrix_accumulator_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_90 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3c203c20u);

    matrix_accumulator_92 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3c203c20u);

    packed_half_128 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_128), 0x3c203c20u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_9),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_fragment_85 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_85),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_91), 0x3c203c20u);

    matrix_accumulator_93 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3c203c20u);

    packed_half_129 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_129), 0x3c203c20u);

    matrix_accumulator_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_21 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x3e473e47u);

    packed_half_149 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_149), 0x3e473e47u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0x3c203c20u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0x3c203c20u);

    matrix_accumulator_79 = (matrix_accumulator_147 << 0x5u) + 0x7ff88000u;

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_accumulator_90 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3e473e47u);

    matrix_accumulator_92 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3e473e47u);

    packed_half_128 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_128), 0x3e473e47u);

    matrix_accumulator_147 = (packed_half_13 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_146 = (matrix_fragment_84 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    packed_half_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_91), 0x3e473e47u);

    matrix_accumulator_93 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3e473e47u);

    packed_half_129 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_129), 0x3e473e47u);

    packed_half_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_104),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_72 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x3c203c20u);

    packed_half_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_105),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3c203c20u);

    packed_half_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_106),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0x3e473e47u);

    packed_half_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_107),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0x3e473e47u);

    matrix_accumulator_77 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = (matrix_fragment_21 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 = (packed_half_149 << 0x5u) + 0x7ff88000u;

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x3c203c20u);

    matrix_fragment_85 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3c203c20u);

    matrix_fragment_5 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                       select_half_operand<2, false, false>(matrix_accumulator_146));

    matrix_accumulator_73 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3c203c20u);

    matrix_accumulator_75 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3c203c20u);

    packed_half_158 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    packed_half_157 = (matrix_accumulator_90 << 0x5u) + 0x7ff88000u;

    packed_half_154 = (matrix_accumulator_92 << 0x5u) + 0x7ff88000u;

    packed_half_153 = (packed_half_128 << 0x5u) + 0x7ff88000u;

    packed_half_159 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    packed_half_156 = (packed_half_91 << 0x5u) + 0x7ff88000u;

    packed_half_155 = (matrix_accumulator_93 << 0x5u) + 0x7ff88000u;

    packed_half_152 = (packed_half_129 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_72 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x3e473e47u);

    matrix_accumulator_74 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3e473e47u);

    matrix_fragment_20 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_accumulator_79));

    matrix_fragment_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x3e473e47u);

    matrix_fragment_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    packed_half_148 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85), 0x3e473e47u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_145 = (packed_half_109 << 0x5u) + 0x7ff88000u;

    packed_half_144 = (packed_half_111 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_73 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3e473e47u);

    matrix_accumulator_75 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3e473e47u);

    packed_half_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3c203c20u);

    packed_half_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x3c203c20u);

    packed_half_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3c203c20u);

    packed_half_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3c203c20u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_158),
                                       select_half_operand<2, false, false>(packed_half_157));

    packed_half_137 = (matrix_accumulator_72 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(packed_half_153));

    packed_half_134 = (matrix_accumulator_74 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_159),
                                       select_half_operand<2, false, false>(packed_half_156));

    packed_half_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_104), 0x3e473e47u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                       select_half_operand<2, false, false>(packed_half_152));

    packed_half_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_105), 0x3e473e47u);

    matrix_fragment_20 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_106), 0x3e473e47u);

    matrix_fragment_21 = add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                        select_half_operand<2, false, false>(packed_half_144));

    packed_half_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_107), 0x3e473e47u);

    packed_half_149 = (matrix_fragment_9 << 0x5u) + 0x7ff88000u;

    packed_half_148 = (packed_half_148 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_136 = (matrix_accumulator_73 << 0x5u) + 0x7ff88000u;

    packed_half_129 = (matrix_accumulator_75 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_128 = (packed_half_104 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                       select_half_operand<2, false, false>(packed_half_134));

    packed_half_110 = (packed_half_105 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_135 = (packed_half_106 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(packed_half_148));

    packed_half_111 = (packed_half_107 << 0x5u) + 0x7ff88000u;

    packed_half_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                    select_half_operand<2, false, false>(packed_half_129));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                       select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_8 = matrix_fragment_12 * 0x4u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    condition_0 = (matrix_fragment_12 & 0x1u) != 0;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    condition_1 = (matrix_fragment_12 & 0x2u) != 0;

    packed_half_13 = matrix_fragment_8 & 0x1cu;

    matrix_fragment_6 = condition_0 ? matrix_fragment_5 : matrix_fragment_10;

    matrix_fragment_9 = condition_0 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_5 = condition_0 ? matrix_fragment_10 : matrix_fragment_5;

    matrix_fragment_4 = condition_0 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_10 =
        uint32_t(int64_t(int32_t(matrix_fragment_12)) * (1ll << 0x1du) >> 32) + packed_half_13;

    packed_half_13 = !condition_1 ? matrix_fragment_6 : matrix_fragment_9;

    matrix_fragment_21 = !condition_1 ? matrix_fragment_9 : matrix_fragment_6;

    matrix_fragment_6 = !condition_1 ? matrix_fragment_5 : matrix_fragment_4;

    matrix_fragment_9 = 0x1u ^ matrix_fragment_10;

    matrix_fragment_4 = !condition_1 ? matrix_fragment_4 : matrix_fragment_5;

    matrix_fragment_8 = 0x2u ^ matrix_fragment_10;

    matrix_fragment_7 = 0x3u ^ matrix_fragment_10;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_13), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        packed_half_13 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_6 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_21), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_21 = result_word;
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

    matrix_fragment_5 = !condition_2 ? packed_half_13 : matrix_fragment_6;

    matrix_fragment_20 = !condition_2 ? matrix_fragment_6 : packed_half_13;

    matrix_accumulator_72 = !condition_2 ? matrix_fragment_21 : matrix_fragment_4;

    packed_half_13 = !condition_2 ? matrix_fragment_4 : matrix_fragment_21;

    matrix_fragment_6 = !condition_3 ? matrix_fragment_5 : matrix_accumulator_72;

    matrix_fragment_21 = !condition_3 ? matrix_fragment_20 : packed_half_13;

    matrix_fragment_5 = !condition_3 ? matrix_accumulator_72 : matrix_fragment_5;

    packed_half_13 = !condition_3 ? packed_half_13 : matrix_fragment_20;

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    packed_half_3 = transpose_half_tile(packed_half_3);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_21));

    packed_half_2 = transpose_half_tile(packed_half_2);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_133 = transpose_half_tile(matrix_accumulator_133);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(packed_half_13));

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    matrix_fragment_6 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_6),
                                       select_half_operand<1, false, false>(matrix_fragment_6));

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4c70ull, matrix_accumulator_72,
        matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5410u);

    matrix_fragment_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6), 0x4100410u);

    packed_half_13 = half_to_float_bits((matrix_fragment_6 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_90, packed_half_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    packed_half_13 = reciprocal_bits(packed_half_13);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    multiply_accumulate_half(packed_half_104, packed_half_105, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_60, matrix_accumulator_61, 0u, 0u);

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_62, matrix_accumulator_63, 0u, 0u);

    multiply_accumulate_half(packed_half_106, packed_half_107, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    matrix_fragment_16 = half_to_float_bits((matrix_fragment_6 >> 16) * 65537u);

    matrix_fragment_6 = matrix_fragment_12;

    matrix_fragment_16 = reciprocal_bits(matrix_fragment_16);

    matrix_fragment_4 = uint32_t(int32_t(matrix_fragment_6) >> 0x2u);

    matrix_fragment_5 = matrix_fragment_4 + 0x8u;

    matrix_fragment_18 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);

    matrix_fragment_18 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x5u >> 32) + matrix_fragment_5;

    packed_half_165 = pack_half(matrix_fragment_16, packed_half_13);

    packed_half_13 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    matrix_fragment_18 = matrix_fragment_18 & 0xffffffe0u;

    packed_half_13 = uint32_t(combine_words(packed_half_13, 0u) << 0x5u >> 32) + matrix_fragment_4;

    matrix_fragment_18 = matrix_fragment_5 + (0u - matrix_fragment_18);

    packed_half_13 = packed_half_13 & 0xffffffe0u;

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_21 = transpose_half_tile(matrix_accumulator_132);

    matrix_fragment_4 = matrix_fragment_4 + (0u - packed_half_13);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(matrix_fragment_18), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        packed_half_2 = result_word;
    }

    packed_half_3 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_150, packed_half_151) + 0x5070ull, memory_ranges, error_counts);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(0u));

    packed_half_13 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_140, matrix_accumulator_141) + 0x5070ull, memory_ranges,
        error_counts);

    packed_half_150 = permute_signed(matrix_fragment_18, matrix_fragment_18, 0x5410u);

    matrix_fragment_18 = transpose_half_tile(matrix_accumulator_126);

    packed_half_151 = permute_signed(packed_half_2, packed_half_2, 0x5410u);

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(packed_half_150));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_accumulator_79 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                            select_half_operand<2, false, false>(packed_half_150));

    packed_half_2 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, packed_half_161) + 0x5070ull, memory_ranges, error_counts);

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(packed_half_150));

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_76),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(0u));

    packed_half_0 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_0),
                                        select_half_operand<2, false, false>(packed_half_151));

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_78 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_78),
                            select_half_operand<2, false, false>(packed_half_151));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_0),
                                          select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_78 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                      select_half_operand<2, false, false>(matrix_accumulator_81));

    packed_half_0 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_138, matrix_accumulator_139) + 0x5070ull, memory_ranges,
        error_counts);

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_accumulator_81));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4e70ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    packed_half_113 = transpose_half_tile(packed_half_113);

    matrix_fragment_15 = transpose_half_tile(matrix_fragment_15);

    matrix_fragment_19 = transpose_half_tile(packed_half_112);

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_14);

    matrix_accumulator_95 = transpose_half_tile(matrix_accumulator_95);

    packed_half_99 = transpose_half_tile(packed_half_99);

    matrix_accumulator_97 = transpose_half_tile(matrix_accumulator_97);

    matrix_accumulator_94 = transpose_half_tile(matrix_accumulator_94);

    matrix_fragment_17 = transpose_half_tile(packed_half_98);

    matrix_accumulator_96 = transpose_half_tile(matrix_accumulator_96);

    matrix_accumulator_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                            select_half_operand<2, false, false>(packed_half_150));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_77 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                                       select_half_operand<2, false, false>(0u));

    packed_half_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_99),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_94 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(packed_half_98));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_95 = uint32_t(int32_t(matrix_accumulator_94) >> 0x1fu);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_accumulator_95 =
        uint32_t(combine_words(matrix_accumulator_95, 0u) << 0x5u >> 32) + matrix_accumulator_94;

    matrix_accumulator_95 = matrix_accumulator_95 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_accumulator_94 = matrix_accumulator_94 + (0u - matrix_accumulator_95);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(matrix_accumulator_94), "r"(0x1fu), "r"(__activemask()));
        packed_half_161 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_accumulator_76,
                             matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_78 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_accumulator_79 = uint32_t(int32_t(matrix_accumulator_78) >> 0x1fu);

    matrix_accumulator_79 =
        uint32_t(combine_words(matrix_accumulator_79, 0u) << 0x5u >> 32) + matrix_accumulator_78;

    matrix_accumulator_79 = matrix_accumulator_79 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_60, matrix_accumulator_61, 0u, 0u);

    packed_half_164 = matrix_accumulator_78 + (0u - matrix_accumulator_79);

    packed_half_109 = transpose_half_tile(packed_half_109);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_165), "r"(packed_half_164), "r"(0x1fu), "r"(__activemask()));
        packed_half_164 = result_word;
    }

    multiply_accumulate_half(packed_half_98, packed_half_99, matrix_fragment_68, matrix_fragment_69,
                             matrix_fragment_70, matrix_fragment_71, packed_half_62,
                             matrix_accumulator_63, 0u, 0u);

    packed_half_108 = transpose_half_tile(packed_half_108);

    packed_half_161 = permute_signed(packed_half_161, packed_half_161, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_64, matrix_accumulator_65, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_accumulator_66, matrix_accumulator_67, 0u, 0u);

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                          select_half_operand<2, false, false>(packed_half_161));

    packed_half_160 = permute_signed(packed_half_164, packed_half_164, 0x5410u);

    packed_half_105 = transpose_half_tile(packed_half_105);

    packed_half_152 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                    select_half_operand<2, false, false>(0u));

    packed_half_107 = transpose_half_tile(packed_half_107);

    packed_half_109 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                    select_half_operand<2, false, false>(0u));

    packed_half_99 = transpose_half_tile(packed_half_99);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_accumulator_141 = transpose_half_tile(matrix_accumulator_141);

    matrix_fragment_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                             select_half_operand<2, false, false>(packed_half_161));

    matrix_accumulator_139 = transpose_half_tile(matrix_accumulator_139);

    packed_half_157 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                                          select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_131 = transpose_half_tile(matrix_accumulator_131);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_152));

    matrix_accumulator_79 = transpose_half_tile(matrix_accumulator_79);

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_156 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                                          select_half_operand<2, false, false>(packed_half_161));

    packed_half_152 = transpose_half_tile(packed_half_98);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    packed_half_104 = transpose_half_tile(packed_half_104);

    packed_half_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                   select_half_operand<2, false, false>(packed_half_109));

    packed_half_106 = transpose_half_tile(packed_half_106);

    matrix_fragment_52 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_158),
                                             select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_140 = transpose_half_tile(matrix_accumulator_140);

    matrix_fragment_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                                             select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_138 = transpose_half_tile(matrix_accumulator_138);

    matrix_fragment_53 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_159),
                                             select_half_operand<2, false, false>(packed_half_161));

    packed_half_108 = transpose_half_tile(matrix_accumulator_130);

    matrix_accumulator_78 = transpose_half_tile(matrix_accumulator_78);

    packed_half_157 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                    select_half_operand<2, false, false>(0u));

    packed_half_153 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                    select_half_operand<2, false, false>(0u));

    packed_half_156 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_156),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(packed_half_156));

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(packed_half_150));

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                          select_half_operand<2, false, false>(packed_half_150));

    packed_half_142 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                          select_half_operand<2, false, false>(packed_half_151));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_accumulator_146 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(packed_half_151));

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_105),
                                    select_half_operand<2, false, false>(0u));

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_150));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_107),
                                    select_half_operand<2, false, false>(0u));

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                          select_half_operand<2, false, false>(packed_half_150));

    packed_half_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_99),
                                   select_half_operand<2, false, false>(0u));

    packed_half_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(packed_half_151));

    packed_half_144 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_147 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(packed_half_151));

    packed_half_148 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_141 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_79 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                      select_half_operand<2, false, false>(0u));

    packed_half_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_104),
                                    select_half_operand<2, false, false>(packed_half_105));

    packed_half_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_106),
                                    select_half_operand<2, false, false>(packed_half_107));

    packed_half_99 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                   select_half_operand<2, false, false>(packed_half_99));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    packed_half_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                    select_half_operand<2, false, false>(matrix_accumulator_141));

    packed_half_107 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                    select_half_operand<2, false, false>(matrix_accumulator_139));

    packed_half_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                    select_half_operand<2, false, false>(matrix_accumulator_131));

    packed_half_109 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                    select_half_operand<2, false, false>(matrix_accumulator_79));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_146),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(packed_half_144));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_147),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_134 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_134),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_135 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_135),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_129 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(packed_half_161));

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<2, false, false>(packed_half_161));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_98, packed_half_99, matrix_accumulator_126,
                             matrix_accumulator_127);

    packed_half_137 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_137),
                                          select_half_operand<2, false, false>(packed_half_160));

    packed_half_135 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_135),
                                    select_half_operand<2, false, false>(0u));

    packed_half_128 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_128),
                                          select_half_operand<2, false, false>(packed_half_160));

    matrix_accumulator_78 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_129),
                                          select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_104, packed_half_105, matrix_accumulator_132,
                             matrix_accumulator_133);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_106, packed_half_107, packed_half_112, packed_half_113);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_108, packed_half_109, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_161));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                             select_half_operand<2, false, false>(packed_half_161));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_137),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_128),
                                       select_half_operand<2, false, false>(packed_half_135));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_110));

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_98, packed_half_99, packed_half_62, matrix_accumulator_63);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_108, packed_half_109, matrix_accumulator_66,
                             matrix_accumulator_67);

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_104, packed_half_105, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_106, packed_half_107, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_68,
                             matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_76),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_66),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                            select_half_operand<2, false, false>(packed_half_91));

    packed_half_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_77),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_68 = multiply_half_pairs(select_half_operand<1, false, false>(packed_half_111),
                                             select_half_operand<1, false, false>(packed_half_111));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_76 =
        multiply_half_pairs(select_half_operand<0, false, false>(packed_half_111),
                            select_half_operand<0, false, false>(packed_half_111));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                      select_half_operand<2, false, false>(matrix_fragment_85),
                                      select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_fragment_87),
                                      select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_78),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_68 =
        fused_multiply_add_half_pairs(select_half_operand<1, false, false>(matrix_accumulator_95),
                                      select_half_operand<1, false, false>(matrix_accumulator_95),
                                      select_half_operand<0, false, false>(matrix_fragment_68));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_79),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_accumulator_76 =
        fused_multiply_add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_95),
                                      select_half_operand<0, false, false>(matrix_accumulator_95),
                                      select_half_operand<0, false, false>(matrix_accumulator_76));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_67 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                       select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(packed_half_62));

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_fragment_69 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                             select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_71 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                             select_half_operand<2, false, false>(packed_half_110));

    matrix_accumulator_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_76 = permute_signed(matrix_accumulator_76, matrix_fragment_68, 0x5410u);

    matrix_accumulator_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(matrix_fragment_84),
                                      select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_fragment_86),
                                      select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                      select_half_operand<2, false, false>(matrix_accumulator_94),
                                      select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_66));

    matrix_fragment_70 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                       select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                                        select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_69 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_67), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_66 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_71), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_76 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_77 = result_word;
    }

    uniform_address_word_6 = blockIdx.y;

    matrix_accumulator_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_13));

    packed_half_62 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                         select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_101),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                            select_half_operand<2, false, false>(packed_half_2));

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                          select_half_operand<2, false, false>(packed_half_0));

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                          select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_72, matrix_accumulator_73, matrix_accumulator_60,
                             matrix_accumulator_61);

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(packed_half_2));

    uniform_index_word_20 =
        uint32_t(combine_words(uniform_index_word_20, 0u) << 0x2u >> 32) + uniform_index_word_16;

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_accumulator_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_accumulator_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                       select_half_operand<2, false, false>(matrix_fragment_69));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_74, matrix_accumulator_75, packed_half_62,
                             matrix_accumulator_63);

    uniform_index_word_20 = uint32_t(int32_t(uniform_index_word_20) >> 0x2u);

    uniform_index_word_13 = (uniform_address_word_6 << 0x3u) + uniform_index_word_13;

    uniform_condition_1 = int32_t(uniform_index_word_11) >= int32_t(uniform_index_word_20);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_64,
                             matrix_accumulator_65);

    uniform_condition_1 =
        int32_t(uniform_index_word_13) > int32_t(0u - 0x4u) && !uniform_condition_1;

    uniform_index_word_21 = uint32_t(combine_words(uniform_index_word_21, 0u) << 0x2u >> 32) +
                            uniform_parameter_word_17;

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_82, matrix_accumulator_83, packed_half_114,
                             packed_half_115);

    matrix_fragment_55 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                             select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_54 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                                             select_half_operand<2, false, false>(packed_half_0));

    packed_half_119 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                                     select_half_operand<2, false, false>(matrix_accumulator_66));

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                                     select_half_operand<2, false, false>(matrix_accumulator_76));

    packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                     select_half_operand<2, false, false>(matrix_accumulator_77));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_120,
                             matrix_accumulator_121);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_121 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_119), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_120 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_115), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_118 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_114), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_72, matrix_accumulator_73, matrix_accumulator_122,
                             matrix_accumulator_123);

    uniform_condition_2 = uniform_condition_1 & uniform_condition_6;

    uniform_index_word_21 = uint32_t(int32_t(uniform_index_word_21) >> 0x2u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_74, matrix_accumulator_75, matrix_accumulator_102,
                             matrix_accumulator_103);

    uniform_address_word_6 = uniform_index_word_11 * uniform_index_word_21 + uniform_index_word_12;

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_54,
                             matrix_fragment_55);

    uniform_address_word_6 = uniform_address_word_6 << (0x7u & 31u);

    split_address_words(uniform_address_word_14, uniform_address_word_15,
                        int64_t(int32_t(uniform_address_word_6)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_14, uniform_address_word_15));
    if (uniform_condition_2) {
        matrix_accumulator_67 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                              select_half_operand<2, false, false>(0u));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_58, matrix_fragment_59,
                            combine_words(uniform_address_word_14, uniform_address_word_15));

        packed_half_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

        matrix_accumulator_63 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_53 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(matrix_fragment_54));

        split_address_words(matrix_fragment_58, matrix_fragment_59,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_58, matrix_fragment_59));

        matrix_fragment_55 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                          select_half_operand<2, false, false>(packed_half_62));

        matrix_fragment_52 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                          select_half_operand<2, false, false>(matrix_accumulator_67));

        matrix_fragment_54 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                          select_half_operand<2, false, false>(matrix_accumulator_63));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_58, matrix_fragment_59), matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);
    }
    uniform_condition_1 = uniform_condition_1 & uniform_condition_0;

    split_address_words(matrix_accumulator_122, matrix_accumulator_123,
                        combine_words(uniform_address_word_14, uniform_address_word_15));

    matrix_accumulator_72 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_72),
                      select_half_operand<2, false, false>(0u));

    uniform_condition_2 = int32_t(uniform_index_word_23) != int32_t(0u);

    matrix_accumulator_73 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_73),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_accumulator_121));

    condition_4 = uniform_condition_1;

    packed_half_119 = add_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                     select_half_operand<2, false, false>(matrix_accumulator_120));

    uniform_address_word_6 = parameters[parameter_output_low_2];
    uniform_address_word_7 = parameters[parameter_output_high_3];

    if (condition_4) {
        matrix_accumulator_103 = permute_signed(matrix_accumulator_81, matrix_fragment_54, 0x5410u);

        split_address_words(matrix_accumulator_122, matrix_accumulator_123,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_122, matrix_accumulator_123));

        matrix_accumulator_102 = permute_signed(matrix_accumulator_80, matrix_fragment_55, 0x5410u);

        packed_half_101 = permute_signed(matrix_accumulator_73, matrix_fragment_52, 0x5410u);

        packed_half_100 = permute_signed(matrix_accumulator_72, matrix_fragment_53, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_122, matrix_accumulator_123) + 0x200ull,
            packed_half_100, packed_half_101, matrix_accumulator_102, matrix_accumulator_103,
            memory_ranges, error_counts);
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3c60ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x3e60ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4060ull, matrix_accumulator_60,
        matrix_accumulator_61, packed_half_62, matrix_accumulator_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4260ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_accumulator_66, matrix_accumulator_67, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4460ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4660ull, matrix_accumulator_72,
        matrix_accumulator_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4860ull, matrix_accumulator_76,
        matrix_accumulator_77, matrix_accumulator_78, matrix_accumulator_79, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4a60ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    packed_half_101 = permute_signed(matrix_accumulator_116, matrix_accumulator_116, 0x5432u);

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                     select_half_operand<2, false, false>(packed_half_118));

    matrix_accumulator_102 = permute_signed(packed_half_119, packed_half_119, 0x5432u);

    packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                     select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(packed_half_101));

    packed_half_119 = add_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                     select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_116 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x4100410u);

    packed_half_119 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_119), 0x4100410u);

    packed_half_100 = half_to_float_bits((matrix_accumulator_116 & 65535u) * 65537u);

    matrix_accumulator_102 = half_to_float_bits((packed_half_119 & 65535u) * 65537u);

    packed_half_119 = half_to_float_bits((packed_half_119 >> 16) * 65537u);

    matrix_accumulator_116 = half_to_float_bits((matrix_accumulator_116 >> 16) * 65537u);

    packed_half_100 = rsqrt_bits(packed_half_100);

    packed_half_101 = rsqrt_bits(matrix_accumulator_116);

    matrix_accumulator_102 = rsqrt_bits(matrix_accumulator_102);

    packed_half_119 = rsqrt_bits(packed_half_119);

    packed_half_101 = pack_half(packed_half_101, packed_half_100);

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_103 = pack_half(packed_half_119, matrix_accumulator_102);

    matrix_fragment_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_84),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_accumulator_90 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                      select_half_operand<2, false, false>(0u));

    packed_half_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_fragment_84 = permute_signed(matrix_fragment_84, packed_half_101, 0x5410u);

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_91 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_91),
                                         select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_85),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_92 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_85 = permute_signed(matrix_fragment_85, matrix_fragment_86, 0x5410u);

    matrix_fragment_86 = permute_signed(matrix_accumulator_90, matrix_accumulator_103, 0x5410u);

    matrix_fragment_87 = permute_signed(matrix_fragment_87, matrix_accumulator_92, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_90, packed_half_91, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_52,
                             matrix_fragment_53);

    matrix_fragment_52 = permute_signed(packed_half_115, packed_half_115, 0x5432u);

    matrix_fragment_53 = permute_signed(packed_half_114, packed_half_114, 0x5432u);

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                     select_half_operand<2, false, false>(matrix_fragment_52));

    packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                     select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_54,
                             matrix_fragment_55);

    packed_half_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0x4100410u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_114), 0x4100410u);

    matrix_fragment_52 = half_to_float_bits((packed_half_115 & 65535u) * 65537u);

    packed_half_115 = half_to_float_bits((packed_half_115 >> 16) * 65537u);

    matrix_fragment_53 = half_to_float_bits((packed_half_114 & 65535u) * 65537u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_114 = half_to_float_bits((packed_half_114 >> 16) * 65537u);

    matrix_fragment_52 = rsqrt_bits(matrix_fragment_52);

    packed_half_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_91),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_56,
                             matrix_fragment_57);

    matrix_accumulator_90 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3c203c20u);

    packed_half_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_91), 0x3c203c20u);

    packed_half_115 = rsqrt_bits(packed_half_115);

    matrix_accumulator_90 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3e473e47u);

    packed_half_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_91), 0x3e473e47u);

    matrix_accumulator_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_53 = rsqrt_bits(matrix_fragment_53);

    matrix_accumulator_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_accumulator_92 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3c203c20u);

    matrix_accumulator_93 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3c203c20u);

    packed_half_114 = rsqrt_bits(packed_half_114);

    matrix_accumulator_92 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3e473e47u);

    matrix_fragment_55 = pack_half(packed_half_115, matrix_fragment_52);

    matrix_accumulator_93 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3e473e47u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_fragment_55));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_60,
                             matrix_accumulator_61);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_52 = pack_half(packed_half_114, matrix_fragment_53);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                          select_half_operand<0, false, false>(packed_half_11));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                          select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    multiply_accumulate_half(packed_half_62, matrix_accumulator_63, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_26, matrix_fragment_27, packed_half_62,
                             matrix_accumulator_63);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                          select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_54 = permute_signed(matrix_fragment_54, matrix_fragment_55, 0x5410u);

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_53, 0x5410u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                          select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    packed_half_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    packed_half_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_62),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_66, matrix_accumulator_67, matrix_fragment_84,
                             matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_66,
                             matrix_accumulator_67);

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    packed_half_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x3c203c20u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_53 = permute_signed(matrix_accumulator_95, matrix_fragment_84, 0x5410u);

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_55 = permute_signed(packed_half_111, matrix_fragment_86, 0x5410u);

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    packed_half_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_62), 0x3e473e47u);

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_68,
                             matrix_fragment_69);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_accumulator_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_accumulator_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_37 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_70,
                             matrix_fragment_71);

    matrix_accumulator_67 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3c203c20u);

    matrix_accumulator_66 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3c203c20u);

    matrix_accumulator_67 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_67), 0x3e473e47u);

    matrix_fragment_35 = (matrix_accumulator_92 << 0x5u) + 0x7ff88000u;

    matrix_fragment_34 = (matrix_accumulator_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_68),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_69),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_72, matrix_accumulator_73, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_72,
                             matrix_accumulator_73);

    matrix_fragment_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3c203c20u);

    matrix_fragment_69 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3c203c20u);

    matrix_fragment_68 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68), 0x3e473e47u);

    matrix_fragment_69 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69), 0x3e473e47u);

    matrix_fragment_32 = (matrix_accumulator_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (packed_half_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_70),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_71),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_fragment_70 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3c203c20u);

    matrix_fragment_71 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3c203c20u);

    matrix_fragment_70 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70), 0x3e473e47u);

    matrix_fragment_71 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71), 0x3e473e47u);

    matrix_accumulator_72 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_59 = (matrix_fragment_71 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_73 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_76, matrix_accumulator_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_76,
                             matrix_accumulator_77);

    matrix_accumulator_72 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x3c203c20u);

    matrix_accumulator_73 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3c203c20u);

    matrix_accumulator_72 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_72), 0x3e473e47u);

    matrix_accumulator_73 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_73), 0x3e473e47u);

    matrix_fragment_28 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = (matrix_fragment_70 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_74 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_30 = (matrix_accumulator_72 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_75 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_accumulator_74 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3c203c20u);

    matrix_accumulator_75 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3c203c20u);

    matrix_accumulator_74 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x3e473e47u);

    matrix_accumulator_75 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x3e473e47u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_31 = (matrix_accumulator_74 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    multiply_accumulate_half(matrix_accumulator_78, matrix_accumulator_79, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_78,
                             matrix_accumulator_79);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_76 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3c203c20u);

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_accumulator_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_76 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_76), 0x3e473e47u);

    matrix_fragment_68 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = (matrix_fragment_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                        select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_accumulator_77 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3c203c20u);

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                        select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_77 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_77), 0x3e473e47u);

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_22 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_23 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = (packed_half_62 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_fragment_52 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_60 = (matrix_accumulator_73 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_78 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3c203c20u);

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                        select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_54 = (matrix_accumulator_76 << 0x5u) + 0x7ff88000u;

    matrix_fragment_71 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_78 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_78), 0x3e473e47u);

    matrix_accumulator_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_79 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3c203c20u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4c70ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_accumulator_61 = (matrix_accumulator_75 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_63 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_55 = (matrix_accumulator_78 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_79 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_79), 0x3e473e47u);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<1, false, false>(packed_half_11), 0x3d343d34u);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    packed_half_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                    select_half_operand<2, false, false>(matrix_fragment_55));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_162, address_word_163) + 0x4e70ull, packed_half_160,
        packed_half_161, address_word_162, address_word_163, memory_ranges, error_counts);

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_fragment_69 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    matrix_accumulator_75 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                           select_half_operand<2, false, false>(packed_half_62));

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_accumulator_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_68 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_66), 0x3e473e47u);

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    packed_half_11 = (matrix_accumulator_77 << 0x5u) + 0x7ff88000u;

    packed_half_62 = (matrix_accumulator_79 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_63 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_66 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_74 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                           select_half_operand<2, false, false>(packed_half_62));

    matrix_fragment_69 = (matrix_accumulator_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_68 = (matrix_fragment_68 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_64 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_67 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    matrix_fragment_70 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_72),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_accumulator_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                       select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_accumulator_74 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                       select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_accumulator_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_accumulator_78 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                       select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_71 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                       select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_accumulator_74 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_73),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_accumulator_75 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                       select_half_operand<2, false, false>(matrix_accumulator_76));

    matrix_accumulator_78 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_77),
                       select_half_operand<2, false, false>(matrix_accumulator_78));

    matrix_accumulator_72 = condition_0 ? matrix_accumulator_74 : matrix_fragment_71;

    matrix_fragment_71 = condition_0 ? matrix_fragment_71 : matrix_accumulator_74;

    matrix_accumulator_73 = condition_0 ? matrix_accumulator_78 : matrix_accumulator_75;

    matrix_accumulator_78 = condition_0 ? matrix_accumulator_75 : matrix_accumulator_78;

    matrix_accumulator_75 = !condition_1 ? matrix_accumulator_72 : matrix_accumulator_73;

    matrix_accumulator_74 = !condition_1 ? matrix_fragment_71 : matrix_accumulator_78;

    matrix_accumulator_73 = !condition_1 ? matrix_accumulator_73 : matrix_accumulator_72;

    matrix_accumulator_78 = !condition_1 ? matrix_accumulator_78 : matrix_fragment_71;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_75), "r"(matrix_fragment_10), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_75 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_74), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_74 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_73), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_73 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_78), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_78 = result_word;
    }

    matrix_fragment_71 = !condition_2 ? matrix_accumulator_75 : matrix_accumulator_74;

    matrix_accumulator_72 = !condition_2 ? matrix_accumulator_74 : matrix_accumulator_75;

    matrix_accumulator_76 = !condition_2 ? matrix_accumulator_73 : matrix_accumulator_78;

    matrix_accumulator_77 = !condition_2 ? matrix_accumulator_78 : matrix_accumulator_73;

    matrix_accumulator_79 = !condition_3 ? matrix_fragment_71 : matrix_accumulator_76;

    matrix_accumulator_80 = !condition_3 ? matrix_accumulator_72 : matrix_accumulator_77;

    matrix_fragment_71 = !condition_3 ? matrix_accumulator_76 : matrix_fragment_71;

    matrix_accumulator_72 = !condition_3 ? matrix_accumulator_77 : matrix_accumulator_72;

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79),
                       select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_accumulator_79 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_79),
                       select_half_operand<1, false, false>(matrix_accumulator_79));

    matrix_accumulator_79 = permute_signed(matrix_accumulator_79, matrix_accumulator_79, 0x5410u);

    matrix_accumulator_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_79), 0x4100410u);

    matrix_fragment_7 = half_to_float_bits((matrix_accumulator_79 & 65535u) * 65537u);

    matrix_accumulator_79 = half_to_float_bits((matrix_accumulator_79 >> 16) * 65537u);

    matrix_fragment_7 = reciprocal_bits(matrix_fragment_7);

    matrix_fragment_10 = uint32_t(int32_t(matrix_fragment_5) >> 0x1fu);

    matrix_fragment_8 = reciprocal_bits(matrix_accumulator_79);

    matrix_fragment_10 =
        uint32_t(combine_words(matrix_fragment_10, 0u) << 0x5u >> 32) + matrix_fragment_5;

    matrix_fragment_10 = matrix_fragment_10 & 0xffffffe0u;

    matrix_fragment_9 = pack_half(matrix_fragment_8, matrix_fragment_7);

    matrix_fragment_8 = matrix_fragment_5 + (0u - matrix_fragment_10);

    matrix_fragment_10 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_6 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_10) >> 0x1fu);

    matrix_fragment_71 = uint32_t(int32_t(matrix_fragment_6) >> 0x1fu);

    matrix_fragment_7 =
        uint32_t(combine_words(matrix_fragment_7, 0u) << 0x5u >> 32) + matrix_fragment_10;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_4 = result_word;
    }

    matrix_fragment_71 =
        uint32_t(combine_words(matrix_fragment_71, 0u) << 0x5u >> 32) + matrix_fragment_6;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_8), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_7 = matrix_fragment_7 & 0xffffffe0u;

    matrix_fragment_71 = matrix_fragment_71 & 0xffffffe0u;

    matrix_accumulator_72 = matrix_fragment_10 + (0u - matrix_fragment_7);

    matrix_accumulator_74 = matrix_fragment_6 + (0u - matrix_fragment_71);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_72), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_72 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_74), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_73 = result_word;
    }

    matrix_fragment_10 = permute_signed(matrix_fragment_4, matrix_fragment_4, 0x5410u);

    matrix_fragment_71 = permute_signed(matrix_fragment_5, matrix_fragment_5, 0x5410u);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_accumulator_72 = permute_signed(matrix_accumulator_72, matrix_accumulator_72, 0x5410u);

    matrix_accumulator_73 = permute_signed(matrix_accumulator_73, matrix_accumulator_73, 0x5410u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_20, matrix_fragment_21, 0u, 0u);

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_63),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    matrix_accumulator_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_67),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    packed_half_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_62),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_18, matrix_fragment_19, 0u, 0u);

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    packed_half_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_62),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                            select_half_operand<2, false, false>(matrix_accumulator_72));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_66),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_accumulator_67 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_67),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                            select_half_operand<2, false, false>(matrix_accumulator_73));

    matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_104, packed_half_105, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_98, packed_half_99, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_106, packed_half_107, matrix_fragment_8,
                             matrix_fragment_9);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_108, packed_half_109, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_11),
                                      select_half_operand<2, false, false>(packed_half_62));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                      select_half_operand<2, false, false>(matrix_accumulator_67));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                      select_half_operand<2, false, false>(matrix_fragment_70));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_98, packed_half_99, matrix_fragment_20,
                             matrix_fragment_21);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_108, packed_half_109, matrix_fragment_14,
                             matrix_fragment_15);

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_104, packed_half_105, matrix_fragment_18,
                             matrix_fragment_19);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_106, packed_half_107, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(packed_half_11));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(packed_half_13));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                   select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_13));

    matrix_fragment_18 = threadIdx.y;

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(packed_half_3));

    packed_half_13 = threadIdx.z;

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = threadIdx.x;

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_0));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(packed_half_0));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_0));

    uniform_block_index_5 = uniform_index_word_11 + 0x1u;

    uniform_condition_1 = int32_t(uniform_block_index_5) >= int32_t(uniform_index_word_20);

    uniform_condition_1 =
        int32_t(uniform_index_word_13) > int32_t(0u - 0x8u) && !uniform_condition_1;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_14,
                             matrix_fragment_15);

    uniform_condition_2 = uniform_condition_1 & uniform_condition_2;

    uniform_index_word_11 = uniform_index_word_11 * uniform_index_word_21 + uniform_index_word_21;

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11, packed_half_160,
                             packed_half_161, matrix_fragment_16, matrix_fragment_17);

    uniform_index_word_12 = uniform_index_word_12 + uniform_index_word_11;

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(packed_half_160, packed_half_161, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_160, packed_half_161,
                             matrix_fragment_44, matrix_fragment_45);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, packed_half_11,
                             address_word_162, address_word_163, matrix_fragment_38,
                             matrix_fragment_39);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             address_word_162, address_word_163, matrix_fragment_42,
                             matrix_fragment_43);

    uniform_index_word_12 = uniform_index_word_12 << (0x7u & 31u);

    condition_0 = (matrix_fragment_18 | packed_half_13 | packed_half_3) != 0;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_12)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));
    if (uniform_condition_2) {
        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(packed_half_2, packed_half_3,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

        packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

        packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(0u));

        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_161),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                          select_half_operand<2, false, false>(packed_half_0));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                          select_half_operand<2, false, false>(packed_half_11));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_160),
                                          select_half_operand<2, false, false>(packed_half_13));

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, packed_half_3), matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_1 & uniform_condition_0;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                      select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                   select_half_operand<2, false, false>(0u));

    packed_half_0 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(0u));

    packed_half_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        split_address_words(packed_half_2, packed_half_3,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_4, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, packed_half_13, 0x5410u);

        split_address_words(matrix_fragment_12, packed_half_13,
                            int64_t(int32_t(matrix_fragment_12)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_5 = permute_signed(matrix_fragment_5, packed_half_0, 0x5410u);

        matrix_fragment_4 = permute_signed(matrix_fragment_24, packed_half_11, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_12, packed_half_13) + 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_block_index_5 = blockIdx.x;

    uniform_address_word_6 = parameters[parameter_publication_optional_low_14];
    uniform_address_word_7 = parameters[parameter_publication_optional_high_15];

    uniform_block_index_5 = uniform_block_index_5 + uniform_reused_word_4;

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_block_index_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));

    split_address_words(packed_half_2, packed_half_3,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(packed_half_2, packed_half_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__ __maxnreg__(127) void phoenix_window_attention_channels_32_attention(
    KernelArguments parameters) {
    run_window_attention_channels_32_attention<false>(parameters.words, nullptr, nullptr);
}
