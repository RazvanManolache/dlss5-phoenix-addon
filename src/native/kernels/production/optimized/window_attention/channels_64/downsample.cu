// DLSS5 Phoenix optimized kernel: window_attention/channels_64/downsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#include "../../../common/boundary_clear.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
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
__device__ __forceinline__ void run_window_attention_channels_64_downsample(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t index_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t address_word_2 = 0;
    uint32_t packed_half_3 = 0;
    uint32_t matrix_fragment_4 = 0;
    uint32_t matrix_fragment_5 = 0;
    uint32_t matrix_fragment_6 = 0;
    uint32_t matrix_fragment_7 = 0;
    uint32_t matrix_fragment_8 = 0;
    uint32_t matrix_fragment_9 = 0;
    uint32_t packed_half_10 = 0;
    uint32_t packed_half_11 = 0;
    uint32_t matrix_fragment_12 = 0;
    uint32_t matrix_fragment_13 = 0;
    uint32_t matrix_fragment_14 = 0;
    uint32_t matrix_fragment_15 = 0;
    uint32_t packed_half_16 = 0;
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
    uint32_t matrix_accumulator_80 = 0;
    uint32_t matrix_accumulator_81 = 0;
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
    uint32_t matrix_accumulator_92 = 0;
    uint32_t matrix_accumulator_93 = 0;
    uint32_t matrix_accumulator_94 = 0;
    uint32_t matrix_accumulator_95 = 0;
    uint32_t matrix_accumulator_96 = 0;
    uint32_t matrix_accumulator_97 = 0;
    uint32_t matrix_fragment_98 = 0;
    uint32_t matrix_fragment_99 = 0;
    uint32_t matrix_accumulator_100 = 0;
    uint32_t matrix_accumulator_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t matrix_fragment_103 = 0;
    uint32_t matrix_fragment_104 = 0;
    uint32_t matrix_fragment_105 = 0;
    uint32_t matrix_accumulator_106 = 0;
    uint32_t matrix_accumulator_107 = 0;
    uint32_t matrix_accumulator_108 = 0;
    uint32_t matrix_accumulator_109 = 0;
    uint32_t matrix_accumulator_110 = 0;
    uint32_t matrix_accumulator_111 = 0;
    uint32_t packed_half_112 = 0;
    uint32_t packed_half_113 = 0;
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
    uint32_t matrix_fragment_130 = 0;
    uint32_t matrix_fragment_131 = 0;
    uint32_t matrix_accumulator_132 = 0;
    uint32_t matrix_accumulator_133 = 0;
    uint32_t matrix_accumulator_134 = 0;
    uint32_t matrix_accumulator_135 = 0;
    uint32_t matrix_accumulator_136 = 0;
    uint32_t matrix_accumulator_137 = 0;
    uint32_t matrix_accumulator_138 = 0;
    uint32_t matrix_accumulator_139 = 0;
    uint32_t matrix_fragment_140 = 0;
    uint32_t matrix_fragment_141 = 0;
    uint32_t matrix_fragment_142 = 0;
    uint32_t matrix_fragment_143 = 0;
    uint32_t matrix_accumulator_144 = 0;
    uint32_t matrix_accumulator_145 = 0;
    uint32_t matrix_accumulator_146 = 0;
    uint32_t matrix_accumulator_147 = 0;
    uint32_t matrix_accumulator_148 = 0;
    uint32_t packed_half_149 = 0;
    uint32_t matrix_fragment_150 = 0;
    uint32_t matrix_fragment_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t packed_half_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t index_word_156 = 0;
    uint32_t address_word_157 = 0;
    uint32_t address_word_158 = 0;
    uint32_t address_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t packed_half_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t packed_half_163 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t packed_half_165 = 0;
    uint32_t uniform_index_word_4 = 0;
    uint32_t uniform_parameter_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_address_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_block_index_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_address_word_15 = 0;
    uint32_t uniform_reused_word_16 = 0;
    uint32_t uniform_reused_word_17 = 0;
    uint32_t uniform_index_word_18 = 0;
    uint32_t uniform_index_word_19 = 0;
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
    uint32_t thread_scratch[64] = {};
    uint64_t async_tokens[2] = {};

    // Block-local operand tile; async tokens below track copies into this storage.
    constexpr uint32_t shared_bytes = DLSS5_TILE_SHARED_BYTES;
    __shared__ __align__(16) uint32_t shared_tile[(shared_bytes + 3) / 4];

    reused_word_1 = 0u;

    uniform_index_word_6 = blockIdx.y;

    uniform_block_index_13 = parameters[parameter_optional_control_15];

    index_word_0 = threadIdx.x;

    uniform_index_word_10 = parameters[parameter_optional_control_10];
    uniform_index_word_11 = parameters[parameter_scalar_control_11];

    matrix_fragment_9 = threadIdx.y;

    uniform_index_word_8 = blockIdx.x;

    uniform_parameter_word_5 = parameters[parameter_tensor_height_8];

    address_word_2 = threadIdx.z;

    uniform_index_word_14 = parameters[parameter_scalar_control_14];

    uniform_index_word_12 = parameters[parameter_tensor_width_9];

    uniform_index_word_4 = min(int32_t(0u), int32_t(uniform_block_index_13));

    uniform_reused_word_16 = 0u;
    uniform_reused_word_17 = 0u;

    uniform_index_word_6 = (uniform_index_word_6 << 0x3u) + uniform_index_word_11;

    uniform_index_word_11 = 0u - uniform_index_word_4 + uniform_parameter_word_5;

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    uniform_condition_2 = uint32_t(uniform_index_word_11) != 0u;

    uniform_index_word_8 = (uniform_index_word_8 << 0x3u) + uniform_index_word_10;

    uniform_index_word_10 = min(int32_t(0u), int32_t(uniform_index_word_14));

    uniform_index_word_7 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x2u >> 32) + uniform_index_word_6;

    uniform_parameter_word_5 = 0u;

    uniform_index_word_14 = uniform_index_word_8 + (0u - uniform_index_word_14);

    uniform_index_word_10 = 0u - uniform_index_word_10 + uniform_index_word_12;

    uniform_index_word_12 = uniform_index_word_6 + (0u - uniform_block_index_13);

    uniform_address_word_9 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_index_word_19 = uniform_index_word_14 + 0x8u;

    uniform_condition_1 = uint32_t(uniform_index_word_10) != 0u;

    if (uniform_condition_2) {
        uniform_index_word_11 = uniform_index_word_11 + 0xffffffffu;
    }

    uniform_index_word_18 = uniform_index_word_12 + 0x8u;

    uniform_block_index_13 = uniform_index_word_12 + 0x7u;

    condition_0 = (address_word_2 | index_word_0 | matrix_fragment_9) != 0;

    if (uniform_condition_2) {
        uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_11) >> 0x1fu);
    }

    uniform_condition_0 = int32_t(uniform_index_word_18) > int32_t(0u);

    uniform_block_index_13 =
        uint32_t(combine_words(uniform_block_index_13, 0u) << 0x1du >> 32) + 0x1u;

    index_word_0 = max(int32_t(0u), int32_t(uniform_index_word_12));

    if (uniform_condition_2) {
        uniform_index_word_6 =
            uint32_t(combine_words(uniform_index_word_6, 0u) << 0x3u >> 32) + uniform_index_word_11;
    }

    uniform_address_word_15 = uniform_index_word_14 + 0x7u;

    uniform_block_index_13 = uniform_condition_0 ? uniform_block_index_13 : 0u;

    if (uniform_condition_1) {
        uniform_index_word_10 = uniform_index_word_10 + 0xffffffffu;
    }

    if (uniform_condition_2) {
        uniform_parameter_word_5 =
            uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    index_word_0 = uint32_t(uint32_t(index_word_0) >> 0x3u);

    uniform_address_word_9 =
        uint32_t(combine_words(uniform_address_word_9, 0u) << 0x2u >> 32) + uniform_index_word_8;

    if (uniform_condition_1) {
        uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);
    }

    uniform_block_index_13 =
        min(int32_t(uniform_block_index_13), int32_t(uniform_parameter_word_5));

    uniform_address_word_15 =
        uint32_t(combine_words(uniform_address_word_15, 0u) << 0x1du >> 32) + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_19) > int32_t(0u);

    if (uniform_condition_1) {
        uniform_index_word_6 =
            uint32_t(combine_words(uniform_index_word_6, 0u) << 0x3u >> 32) + uniform_index_word_10;
    }

    uniform_index_word_4 = 0u;

    uniform_parameter_word_5 = uint32_t(int32_t(uniform_index_word_7) >> 0x2u);

    condition_0 = int32_t(index_word_0) >= int32_t(uniform_block_index_13) || condition_0;

    uniform_address_word_9 = uint32_t(int32_t(uniform_address_word_9) >> 0x2u);

    uniform_address_word_15 = uniform_condition_0 ? uniform_address_word_15 : 0u;

    if (uniform_condition_1) {
        uniform_index_word_4 =
            uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1du) >> 32) + 0x1u;
    }

    address_word_2 = max(int32_t(0u), int32_t(uniform_index_word_14));
    if (!condition_0) {
        matrix_fragment_6 = uniform_index_word_4;

        matrix_fragment_4 = uint32_t(uint32_t(address_word_2) >> 0x3u);

        matrix_fragment_6 = min(int32_t(matrix_fragment_6), int32_t(uniform_address_word_15));
        do {
            condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_6);
            if (!condition_0) {
                matrix_fragment_8 = index_word_0 * uniform_index_word_4;

                matrix_fragment_5 = matrix_fragment_4;
                do {
                    address_word_2 = parameters[parameter_wait_optional_low_12];
                    packed_half_3 = parameters[parameter_wait_optional_high_13];

                    matrix_fragment_7 = matrix_fragment_8 + matrix_fragment_5;

                    split_address_words(address_word_2, packed_half_3,
                                        int64_t(int32_t(matrix_fragment_7)) *
                                                int64_t(int32_t(0x4u)) +
                                            combine_words(address_word_2, packed_half_3));

                    matrix_fragment_7 = pipeline_read<CheckBounds>(
                        combine_words(address_word_2, packed_half_3), memory_ranges, error_counts);

                    matrix_fragment_5 = matrix_fragment_5 + 0x1u;

                    condition_1 = uint32_t(matrix_fragment_5) != uint32_t(matrix_fragment_6);

                    condition_0 = int32_t(matrix_fragment_7) > -1;
                    if (!condition_0) {
                        do {
                            __nanosleep(0x40u);

                            matrix_fragment_7 = pipeline_read<CheckBounds>(
                                combine_words(address_word_2, packed_half_3), memory_ranges,
                                error_counts);

                            condition_0 = int32_t(matrix_fragment_7) > -1;
                        } while (!condition_0);
                    }
                } while (condition_1);
            }
            index_word_0 = index_word_0 + 0x1u;

            condition_0 = uint32_t(index_word_0) != uint32_t(uniform_block_index_13);
        } while (condition_0);
    }
    __syncwarp(__activemask());

    uniform_index_word_6 = parameters[parameter_tensor_height_8];
    uniform_index_word_7 = parameters[parameter_tensor_width_9];

    index_word_0 = matrix_fragment_9 + uniform_parameter_word_5;

    packed_half_10 = uniform_address_word_9;

    condition_4 = false;

    address_word_157 = threadIdx.x & 31u;

    condition_2 = true;

    address_word_158 = parameters[parameter_trained_weights_low_4];
    address_word_159 = parameters[parameter_trained_weights_high_5];

    packed_half_10 = packed_half_10 + 0x1u;

    matrix_fragment_6 = 0u;

    packed_half_11 = 0u;

    matrix_accumulator_17 = 0u;

    uniform_index_word_14 = broadcast_warp_word(packed_half_10, true, uniform_index_word_14);

    uniform_index_word_10 = uniform_index_word_6 & 0xfffffffcu;

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_7) >> 0x1fu);

    uniform_index_word_12 = uniform_index_word_7 & 0xfffffffcu;

    uniform_condition_0 = uint32_t(uniform_index_word_10) != uint32_t(0x4u);

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_11, 0u) << 0x2u >> 32) + uniform_index_word_7;

    matrix_fragment_5 = uniform_index_word_12;

    uniform_index_word_10 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_6;

    __syncthreads();

    condition_0 = uniform_condition_0;

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_10) >> 0x2u);

    uniform_condition_0 = uint32_t(uniform_index_word_12) != uint32_t(0x4u);

    uniform_condition_1 = int32_t(uniform_address_word_9) >= int32_t(uniform_index_word_11);

    uniform_condition_2 = int32_t(uniform_index_word_14) >= int32_t(uniform_index_word_11);

    uniform_condition_1 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x4u) && !uniform_condition_1;

    uniform_condition_2 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x8u) && !uniform_condition_2;

    if (condition_0) {
        condition_1 = int32_t(index_word_0) >= int32_t(uniform_index_word_10);

        condition_3 = int32_t(index_word_0) >= int32_t(uniform_index_word_10);
    }

    condition_5 = uniform_condition_1;

    if (condition_0) {
        matrix_fragment_6 = index_word_0 * uniform_index_word_11;
    }

    condition_6 = uniform_condition_2;

    if (condition_0) {
        packed_half_11 = index_word_0 * uniform_index_word_11;

        condition_4 = int32_t(index_word_0) < int32_t(0u) || condition_1;
    }

    condition_1 = false;

    if (condition_0) {
        matrix_accumulator_17 = index_word_0 * uniform_index_word_11;

        condition_1 = int32_t(index_word_0) < int32_t(0u) || condition_3;
    }

    condition_3 = true;

    address_word_2 = uint32_t(condition_4) << 4;

    if (condition_0) {
        condition_2 = !condition_4;
    }

    condition_4 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_4;

    if (condition_0) {
        condition_3 = !condition_1;
    }

    condition_2 = condition_2 & (condition_4 | condition_5);

    condition_4 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_1;

    condition_5 = uniform_condition_1;

    matrix_fragment_4 = uint32_t(condition_2) << 2;

    condition_3 = condition_3 & (condition_4 | condition_5);

    if (condition_0) {
        condition_5 = int32_t(index_word_0) >= int32_t(uniform_index_word_10);
    }

    condition_2 = int32_t(address_word_2) != int32_t(0u);

    condition_4 = false;

    address_word_2 = uint32_t(condition_1) << 1;

    if (condition_0) {
        condition_4 = int32_t(index_word_0) < int32_t(0u) || condition_5;
    }

    packed_half_3 = uint32_t(condition_2) << 2;

    condition_5 = true;

    condition_2 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_4;

    if (condition_0) {
        condition_5 = !condition_4;
    }

    condition_1 = false;

    condition_5 = condition_5 & (condition_2 | condition_6);

    if (condition_0) {
        condition_6 = int32_t(index_word_0) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_13 = condition_4 ? packed_half_10 : 0u;

    if (condition_0) {
        condition_1 = int32_t(index_word_0) < int32_t(0u) || condition_6;
    }

    condition_6 = true;

    matrix_fragment_8 = uint32_t(condition_1) << 1;

    if (condition_5) {
        matrix_fragment_14 = parameters[parameter_input_low_0];
        matrix_fragment_15 = parameters[parameter_input_high_1];
    }

    if (condition_0) {
        condition_6 = !condition_1;
    }

    condition_2 = uint32_t(matrix_fragment_5) == uint32_t(0x4u) || condition_1;

    condition_1 = uniform_condition_2;

    condition_4 = int32_t(matrix_fragment_8) != int32_t(0u);

    condition_6 = condition_6 & (condition_2 | condition_1);

    if (condition_3) {
        matrix_fragment_8 = parameters[parameter_input_low_0];
        matrix_fragment_9 = parameters[parameter_input_high_1];
    }

    condition_2 = int32_t(packed_half_3) != int32_t(0u);

    condition_1 = int32_t(address_word_2) != int32_t(0u);

    matrix_fragment_5 = !condition_2 ? 0u : uniform_address_word_9;

    condition_2 = int32_t(matrix_fragment_4) != int32_t(0u);

    matrix_fragment_7 = !condition_1 ? 0u : uniform_address_word_9;

    condition_1 = uniform_condition_0;

    matrix_fragment_4 = 0u;

    if (condition_0) {
        matrix_fragment_4 = index_word_0 * uniform_index_word_11;
    }

    if (condition_6) {
        matrix_accumulator_22 = parameters[parameter_input_low_0];
        matrix_accumulator_23 = parameters[parameter_input_high_1];
    }

    index_word_0 = !condition_1 ? matrix_fragment_13 : packed_half_10;

    matrix_fragment_7 = !condition_1 ? matrix_fragment_7 : uniform_address_word_9;

    matrix_fragment_5 = !condition_1 ? matrix_fragment_5 : uniform_address_word_9;

    if (condition_5) {
        packed_half_11 = index_word_0 + packed_half_11;
    }

    if (condition_3) {
        matrix_fragment_6 = matrix_fragment_7 + matrix_fragment_6;
    }

    if (condition_2) {
        address_word_2 = parameters[parameter_input_low_0];
        packed_half_3 = parameters[parameter_input_high_1];

        matrix_fragment_4 = matrix_fragment_5 + matrix_fragment_4;
    }

    if (!condition_1) {
        packed_half_10 = condition_4 ? packed_half_10 : 0u;
    }

    if (condition_5) {
        packed_half_11 = packed_half_11 * 0x100u;
    }

    if (!condition_2) {
        matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (condition_3) {
        matrix_fragment_7 = matrix_fragment_6 * 0x100u;
    }

    if (!condition_3) {
        index_word_0 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                     select_half_operand<2, false, false>(0u));
    }

    if (condition_2) {
        matrix_fragment_5 = matrix_fragment_4 * 0x100u;
    }

    if (condition_6) {
        packed_half_10 = packed_half_10 + matrix_accumulator_17;
    }

    if (condition_5) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(packed_half_11)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_5) {
        packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                       select_half_operand<2, false, false>(0u));
    }

    if (!condition_2) {
        matrix_accumulator_19 =
            permute_signed(matrix_accumulator_19, matrix_accumulator_19, 0x5410u);
    }

    if (condition_3) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_5) {
        packed_half_11 = permute_signed(packed_half_11, packed_half_11, 0x5410u);
    }

    if (!condition_3) {
        matrix_fragment_7 = permute_signed(index_word_0, index_word_0, 0x5410u);
    }

    index_word_0 = address_word_157;

    if (condition_3) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    if (!condition_2) {
        packed_half_16 = matrix_accumulator_19;

        matrix_accumulator_17 = matrix_accumulator_19;

        matrix_accumulator_18 = matrix_accumulator_19;
    }

    if (condition_2) {
        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(matrix_fragment_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(address_word_2, packed_half_3));
    }

    if (!condition_5) {
        matrix_fragment_8 = packed_half_11;

        matrix_fragment_9 = packed_half_11;
    }

    if (condition_6) {
        matrix_fragment_25 = packed_half_10 * 0x100u;
    }

    if (!condition_5) {
        packed_half_10 = packed_half_11;
    }

    if (condition_5) {
        split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_6) {
        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                           select_half_operand<2, false, false>(0u));
    }

    index_word_156 = index_word_0 & 0xfffffffcu;

    if (condition_2) {
        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, packed_half_3));
    }

    if (!condition_3) {
        matrix_fragment_4 = matrix_fragment_7;

        matrix_fragment_5 = matrix_fragment_7;

        matrix_fragment_6 = matrix_fragment_7;
    }

    if (condition_6) {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(matrix_fragment_25)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_22, matrix_accumulator_23));
    }

    if (!condition_6) {
        matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(address_word_2, packed_half_3), packed_half_16, matrix_accumulator_17,
            matrix_accumulator_18, matrix_accumulator_19, memory_ranges, error_counts);
    }

    if (condition_5) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_20, matrix_accumulator_21), matrix_fragment_8,
            matrix_fragment_9, packed_half_10, packed_half_11, memory_ranges, error_counts);
    }

    if (!condition_6) {
        matrix_fragment_14 = matrix_fragment_15;
    }

    if (condition_6) {
        split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_22, matrix_accumulator_23));
    }

    if (condition_3) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13) + 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }

    packed_half_3 = 0u - index_word_156 + address_word_157;

    matrix_accumulator_21 = address_word_157 + 0x4u + (0u - index_word_156);

    split_address_words(address_word_2, packed_half_3,
                        int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_27 = address_word_157 + 0xcu + (0u - index_word_156);

    if (!condition_6) {
        matrix_fragment_12 = matrix_fragment_15;

        matrix_fragment_13 = matrix_fragment_15;
    }

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        uint64_t(uint32_t(matrix_accumulator_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_25 = address_word_157 + 0x8u + (0u - index_word_156);

    matrix_accumulator_109 = load_aligned_word<CheckBounds>(
        combine_words(address_word_2, packed_half_3) + 0x7010ull, memory_ranges, error_counts);

    matrix_fragment_31 = address_word_157 + 0x1cu + (0u - index_word_156);

    if (condition_6) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0x200ull,
            matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
            memory_ranges, error_counts);
    }

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        uint64_t(uint32_t(matrix_fragment_25)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_28 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x7010ull, memory_ranges,
        error_counts);

    packed_half_3 = address_word_157 + 0x14u + (0u - index_word_156);

    matrix_fragment_29 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_24, matrix_fragment_25) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_22, matrix_accumulator_23,
                        uint64_t(uint32_t(matrix_fragment_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_27 = address_word_157 + 0x10u + (0u - index_word_156);

    matrix_accumulator_21 = address_word_157 + 0x18u + (0u - index_word_156);

    split_address_words(address_word_2, packed_half_3,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_30 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_22, matrix_accumulator_23) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_26, matrix_fragment_27,
                        uint64_t(uint32_t(matrix_fragment_27)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_32 = load_aligned_word<CheckBounds>(
        combine_words(address_word_2, packed_half_3) + 0x7010ull, memory_ranges, error_counts);

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        uint64_t(uint32_t(matrix_accumulator_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_27 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_26, matrix_fragment_27) + 0x7010ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        uint64_t(uint32_t(matrix_fragment_31)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_37 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x7010ull, memory_ranges,
        error_counts);

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_24, matrix_fragment_25) + 0x7010ull, memory_ranges,
        error_counts);

    uniform_condition_0 = true;

    packed_half_154 = unpack_fp8_bits(packed_half_16 >> 16);

    packed_half_152 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    packed_half_112 = unpack_fp8_bits(packed_half_11);

    matrix_fragment_151 = unpack_fp8_bits(matrix_accumulator_18);

    matrix_fragment_150 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    matrix_accumulator_148 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_accumulator_147 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    matrix_accumulator_146 = unpack_fp8_bits(matrix_fragment_4);

    matrix_accumulator_145 = unpack_fp8_bits(matrix_fragment_4 >> 16);

    matrix_accumulator_144 = unpack_fp8_bits(matrix_fragment_5);

    matrix_fragment_143 = unpack_fp8_bits(matrix_fragment_5 >> 16);

    matrix_accumulator_123 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_accumulator_119 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_accumulator_118 = unpack_fp8_bits(packed_half_10);

    packed_half_113 = unpack_fp8_bits(packed_half_10 >> 16);

    packed_half_11 = unpack_fp8_bits(packed_half_11 >> 16);

    matrix_fragment_140 = unpack_fp8_bits(matrix_fragment_7);

    matrix_fragment_131 = unpack_fp8_bits(matrix_fragment_7 >> 16);

    matrix_fragment_130 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_5 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    packed_half_155 = unpack_fp8_bits(packed_half_16);

    packed_half_153 = unpack_fp8_bits(matrix_accumulator_17);

    matrix_fragment_142 = unpack_fp8_bits(matrix_fragment_6);

    matrix_fragment_141 = unpack_fp8_bits(matrix_fragment_6 >> 16);

    matrix_accumulator_122 = unpack_fp8_bits(matrix_fragment_9);

    address_word_2 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_7 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    packed_half_3 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_6 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_123),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_fragment_28));

    packed_half_16 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_151),
                                         select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_148),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_118),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_150),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_143),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    packed_half_10 = 0u;

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_130),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_2),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_142),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_140),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_141),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_131),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_122),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                            select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_39));
    do {
        matrix_fragment_103 = packed_half_10 * 0x800u;

        split_address_words(matrix_fragment_102, matrix_fragment_103,
                            uint64_t(uint32_t(matrix_fragment_103)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        split_address_words(matrix_fragment_102, matrix_fragment_103,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_102, matrix_fragment_103));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_102, matrix_fragment_103),
                                            matrix_fragment_68, matrix_fragment_69,
                                            matrix_fragment_70, matrix_fragment_71, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x200ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1000ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x400ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1200ull, matrix_fragment_88,
            matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1400ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        matrix_accumulator_101 = packed_half_10 * 0x400u;

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x600ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_accumulator_101)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1600ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4000ull,
            matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_82,
            matrix_accumulator_83, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4200ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_150),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_147),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_11),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_130),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_151),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_148),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_118),
                          select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                          select_half_operand<2, false, false>(matrix_fragment_75));

        matrix_fragment_73 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_143),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_75 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_131),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_96 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_73 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_144),
                          select_half_operand<2, false, false>(matrix_fragment_73));

        matrix_fragment_75 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_140),
                          select_half_operand<2, false, false>(matrix_fragment_75));

        packed_half_149 = 0x2b28u;

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_68, matrix_fragment_69, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                          select_half_operand<2, false, false>(0u));

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_70, matrix_fragment_71, 0u, 0u);

        matrix_fragment_72 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_146),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_141),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_70 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_68 =
            pack_fp8_pair(select_half_operand<2, false, false>(address_word_2),
                          select_half_operand<2, false, false>(matrix_fragment_69));

        matrix_fragment_74 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_142),
                          select_half_operand<2, false, false>(matrix_fragment_71));

        matrix_fragment_69 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                          select_half_operand<2, false, false>(matrix_fragment_70));

        matrix_fragment_70 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                          select_half_operand<2, false, false>(matrix_accumulator_96));

        matrix_fragment_71 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                          select_half_operand<2, false, false>(matrix_fragment_98));

        multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_56, matrix_fragment_57, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_94,
                                 matrix_accumulator_95);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_54, matrix_fragment_55, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_58, matrix_fragment_59, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_60, matrix_fragment_61, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_56, matrix_fragment_57, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_88, matrix_fragment_89, matrix_fragment_104,
                                 matrix_fragment_105);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_58, matrix_fragment_59, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_54,
                                 matrix_fragment_55);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_44, matrix_fragment_45, matrix_fragment_98,
                                 matrix_fragment_99);

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_96,
                                 matrix_accumulator_97);

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

        matrix_fragment_45 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_61 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_52 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_62, matrix_fragment_63, 0u, 0u);

        matrix_fragment_62 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_62),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_52 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        matrix_fragment_55 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

        matrix_fragment_63 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62),
            select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

        matrix_fragment_54 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

        matrix_fragment_62 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

        matrix_fragment_63 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_63));

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_52),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_54 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_accumulator_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0xc400c400u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_55 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_54),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_53 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52),
            select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

        matrix_fragment_62 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_62), 0x3b283b28u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54),
            select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

        matrix_fragment_56 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                select_half_operand<2, false, false>(matrix_fragment_53));

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_106,
                                 matrix_accumulator_107);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4400ull,
            matrix_fragment_52, matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
            memory_ranges, error_counts);

        matrix_fragment_45 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_accumulator_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_92),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x3b283b28u);

        matrix_fragment_62 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                select_half_operand<2, false, false>(matrix_fragment_62));

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_90, matrix_fragment_91, matrix_accumulator_94,
                                 matrix_accumulator_95);

        matrix_accumulator_93 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_accumulator_93 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                          select_half_operand<2, false, false>(matrix_fragment_61));

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_134,
                                 matrix_accumulator_135);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_115), 0x44004400u);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_116), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_132,
                                 matrix_accumulator_133);

        matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                          select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                          select_half_operand<2, false, false>(matrix_fragment_47));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                          select_half_operand<2, false, false>(matrix_accumulator_93));

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_114), 0x44004400u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x800ull,
            matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
            matrix_accumulator_95, memory_ranges, error_counts);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_80, matrix_accumulator_81, 0u, 0u);

        matrix_accumulator_121 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_117), 0x44004400u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4600ull,
            matrix_fragment_60, matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
            memory_ranges, error_counts);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_121 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_82, matrix_accumulator_83, 0u, 0u);

        matrix_accumulator_120 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_84, matrix_fragment_85, 0u, 0u);

        matrix_accumulator_120 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_accumulator_120), 0x3b283b28u);

        multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_86, matrix_fragment_87, 0u, 0u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_47 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_44), 0x3b283b28u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_121),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_47 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_115 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_121),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        matrix_accumulator_114 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_44 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

        matrix_fragment_46 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_116 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                                select_half_operand<2, false, false>(matrix_accumulator_120));

        matrix_fragment_46 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0xc400c400u);

        matrix_accumulator_117 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_117),
                                select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_accumulator_120 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_44 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_fragment_45));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1800ull, matrix_fragment_56,
            matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges,
            error_counts);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_46),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_44 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0xc400c400u);

        matrix_fragment_46 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        matrix_fragment_45 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_44),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        matrix_accumulator_121 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_fragment_46));

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44),
            select_half_operand<2, false, false>(matrix_fragment_45), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0xa00ull, matrix_fragment_44,
            matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges,
            error_counts);

        matrix_accumulator_125 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_88));

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        matrix_fragment_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(matrix_accumulator_125));

        matrix_accumulator_114 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_117 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_116 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(matrix_accumulator_114));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                          select_half_operand<2, false, false>(matrix_accumulator_117));

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                          select_half_operand<2, false, false>(matrix_accumulator_116));

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_80, matrix_accumulator_81, 0u, 0u);

        matrix_accumulator_80 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

        matrix_accumulator_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_82, matrix_accumulator_83, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_84, matrix_fragment_85, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_86, matrix_fragment_87, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1a00ull, matrix_fragment_88,
            matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4800ull,
            matrix_fragment_84, matrix_fragment_85, matrix_fragment_86, matrix_fragment_87,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_64,
                                 matrix_fragment_65);

        matrix_fragment_64 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_132), 0x44004400u);

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_80),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_81 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0xc400c400u);

        matrix_fragment_65 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80),
            select_half_operand<2, false, false>(matrix_fragment_65), 0x3b283b28u);

        matrix_fragment_64 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0x44004400u);

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_accumulator_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        matrix_accumulator_82 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_81),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81),
            select_half_operand<2, false, false>(matrix_accumulator_82), 0x3b283b28u);

        matrix_accumulator_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        matrix_accumulator_81 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_80),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_82 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82), 0xc400c400u);

        matrix_accumulator_81 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80),
            select_half_operand<2, false, false>(matrix_accumulator_81), 0x3b283b28u);

        matrix_accumulator_80 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99), 0x44004400u);

        matrix_fragment_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        matrix_accumulator_81 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_82),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80), 0xc400c400u);

        matrix_accumulator_81 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82),
            select_half_operand<2, false, false>(matrix_accumulator_81), 0x3b283b28u);

        matrix_accumulator_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_81 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_80),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_64,
                                 matrix_fragment_65);

        matrix_accumulator_82 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82), 0xc400c400u);

        matrix_accumulator_81 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80),
            select_half_operand<2, false, false>(matrix_accumulator_81), 0x3b283b28u);

        matrix_fragment_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                                select_half_operand<2, false, false>(matrix_accumulator_81));

        matrix_accumulator_81 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

        matrix_accumulator_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_82),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_81 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81), 0xc400c400u);

        matrix_accumulator_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82),
            select_half_operand<2, false, false>(matrix_accumulator_80), 0x3b283b28u);

        matrix_accumulator_82 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_133), 0x44004400u);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(matrix_accumulator_80));

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_66,
                                 matrix_fragment_67);

        matrix_accumulator_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_81),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_82 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82), 0xc400c400u);

        matrix_accumulator_80 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81),
            select_half_operand<2, false, false>(matrix_accumulator_80), 0x3b283b28u);

        matrix_fragment_66 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_136), 0x44004400u);

        matrix_fragment_67 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_fragment_66 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66), 0xc400c400u);

        matrix_accumulator_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(matrix_accumulator_80));

        matrix_accumulator_80 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67), 0xc400c400u);

        matrix_accumulator_132 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                select_half_operand<2, false, false>(matrix_accumulator_83));

        matrix_accumulator_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_82),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_66),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_83 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82),
            select_half_operand<2, false, false>(matrix_accumulator_83), 0x3b283b28u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_66),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_80),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_133 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                select_half_operand<2, false, false>(matrix_accumulator_83));

        matrix_fragment_67 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80),
            select_half_operand<2, false, false>(matrix_fragment_67), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4a00ull,
            matrix_accumulator_80, matrix_accumulator_81, matrix_accumulator_82,
            matrix_accumulator_83, memory_ranges, error_counts);

        matrix_accumulator_138 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_137), 0x44004400u);

        matrix_accumulator_139 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_accumulator_138 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_138), 0xc400c400u);

        packed_half_160 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_139), 0xc400c400u);

        matrix_accumulator_139 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_138),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_139 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_138),
            select_half_operand<2, false, false>(matrix_accumulator_139), 0x3b283b28u);

        matrix_accumulator_138 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_160),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_138 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_160),
            select_half_operand<2, false, false>(matrix_accumulator_138), 0x3b283b28u);

        matrix_accumulator_138 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(matrix_accumulator_138));

        matrix_fragment_65 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_67 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_accumulator_134 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_64 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                          select_half_operand<2, false, false>(matrix_fragment_65));

        matrix_fragment_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_76), 0x44004400u);

        matrix_fragment_65 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                          select_half_operand<2, false, false>(matrix_accumulator_134));

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0xc400c400u);

        matrix_fragment_99 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_78), 0x44004400u);

        matrix_accumulator_134 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99), 0xc400c400u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_134),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_135 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134),
            select_half_operand<2, false, false>(matrix_fragment_98), 0x3b283b28u);

        matrix_fragment_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_77), 0x44004400u);

        matrix_accumulator_134 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_79), 0x44004400u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0xc400c400u);

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_accumulator_136 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0xc400c400u);

        matrix_accumulator_134 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_134 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x3b283b28u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_136),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_136),
            select_half_operand<2, false, false>(matrix_fragment_98), 0x3b283b28u);

        matrix_accumulator_135 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78),
                                select_half_operand<2, false, false>(matrix_accumulator_135));

        matrix_fragment_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                                select_half_operand<2, false, false>(matrix_fragment_98));

        matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_78 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_139 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                select_half_operand<2, false, false>(matrix_accumulator_139));

        matrix_accumulator_138 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_134 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                select_half_operand<2, false, false>(matrix_accumulator_134));

        matrix_fragment_77 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_76 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_66 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                          select_half_operand<2, false, false>(matrix_fragment_67));

        matrix_fragment_98 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                          select_half_operand<2, false, false>(matrix_fragment_78));

        matrix_fragment_67 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                          select_half_operand<2, false, false>(matrix_accumulator_138));

        matrix_accumulator_96 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(matrix_fragment_77));

        matrix_accumulator_97 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                          select_half_operand<2, false, false>(matrix_fragment_76));

        matrix_fragment_99 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                          select_half_operand<2, false, false>(matrix_fragment_79));

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_52, matrix_fragment_53, matrix_accumulator_126,
                                 matrix_accumulator_127);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1c00ull, matrix_fragment_76,
            matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98,
                                 matrix_fragment_99, matrix_fragment_52, matrix_fragment_53,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_54, matrix_fragment_55, matrix_fragment_104,
                                 matrix_fragment_105);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98,
                                 matrix_fragment_99, matrix_fragment_54, matrix_fragment_55,
                                 matrix_accumulator_116, matrix_accumulator_117);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0xc00ull, matrix_fragment_52,
            matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_92, matrix_accumulator_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_92, matrix_accumulator_93, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_60, matrix_fragment_61, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_64,
                                 matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_128,
                                 matrix_fragment_129);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0xe00ull, matrix_fragment_64,
            matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98,
                                 matrix_fragment_99, matrix_fragment_60, matrix_fragment_61,
                                 matrix_accumulator_120, matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98,
                                 matrix_fragment_99, matrix_fragment_62, matrix_fragment_63,
                                 matrix_accumulator_124, matrix_accumulator_125);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_102, matrix_fragment_103) + 0x1e00ull, matrix_fragment_60,
            matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges,
            error_counts);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_accumulator_94, matrix_accumulator_95, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_92,
                                 matrix_accumulator_93);

        multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_44, matrix_fragment_45, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_94, matrix_accumulator_95, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_96,
                                 matrix_accumulator_97);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_46, matrix_fragment_47, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_88, matrix_fragment_89, matrix_fragment_56,
                                 matrix_fragment_57);

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_132), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_44,
                                 matrix_fragment_45);

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_133), 0x44004400u);

        matrix_fragment_56 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_94,
                                 matrix_accumulator_95);

        matrix_fragment_58 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x44004400u);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_134,
                                 matrix_accumulator_135);

        matrix_fragment_58 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_56),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_88 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x44004400u);

        matrix_fragment_89 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0xc400c400u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_59 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_88), 0x3b283b28u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_58),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_88 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_89),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_58),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89),
            select_half_operand<2, false, false>(matrix_fragment_88), 0x3b283b28u);

        matrix_fragment_88 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                                select_half_operand<2, false, false>(matrix_fragment_57));

        matrix_fragment_57 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        matrix_fragment_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_56 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_90, matrix_fragment_91, matrix_accumulator_138,
                                 matrix_accumulator_139);

        matrix_accumulator_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                select_half_operand<2, false, false>(matrix_fragment_59));

        matrix_fragment_57 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

        matrix_fragment_59 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        matrix_fragment_91 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_90 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

        matrix_fragment_91 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0xc400c400u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_57),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_59 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

        matrix_accumulator_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_90),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_91),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_56 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_57),
            select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

        matrix_fragment_57 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_59),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_58 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90),
            select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91),
            select_half_operand<2, false, false>(matrix_fragment_98), 0x3b283b28u);

        matrix_fragment_90 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_59),
            select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

        matrix_fragment_44 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_58));

        matrix_fragment_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_98));

        matrix_fragment_47 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_fragment_90));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_46 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_56));

        matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                           select_half_operand<2, false, false>(0u));

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4c00ull,
            matrix_fragment_56, matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
            memory_ranges, error_counts);

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                                           select_half_operand<2, false, false>(0u));

        matrix_accumulator_96 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

        matrix_fragment_89 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                          select_half_operand<2, false, false>(matrix_fragment_90));

        matrix_fragment_90 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_fragment_91));

        matrix_fragment_91 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                          select_half_operand<2, false, false>(matrix_fragment_98));

        matrix_fragment_88 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_accumulator_96 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96), 0xc400c400u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x4e00ull,
            matrix_fragment_44, matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
            memory_ranges, error_counts);

        matrix_fragment_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

        matrix_accumulator_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_96),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_99 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0xc400c400u);

        matrix_fragment_98 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

        matrix_accumulator_97 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_96),
            select_half_operand<2, false, false>(matrix_accumulator_97), 0x3b283b28u);

        matrix_fragment_98 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98), 0xc400c400u);

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_86, matrix_fragment_87, matrix_fragment_104,
                                 matrix_fragment_105);

        matrix_accumulator_101 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

        matrix_accumulator_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_99),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_104 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                select_half_operand<2, false, false>(matrix_accumulator_97));

        matrix_accumulator_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_98),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_accumulator_101 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101), 0xc400c400u);

        matrix_accumulator_96 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99),
            select_half_operand<2, false, false>(matrix_accumulator_96), 0x3b283b28u);

        matrix_accumulator_92 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_98),
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x3b283b28u);

        matrix_fragment_99 = packed_half_10 * 0x200u;

        packed_half_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_101),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_105 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                select_half_operand<2, false, false>(matrix_accumulator_92));

        matrix_accumulator_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_138), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_80, matrix_accumulator_81,
                                 matrix_accumulator_106, matrix_accumulator_107);

        matrix_accumulator_97 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_134), 0x44004400u);

        packed_half_162 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0xc400c400u);

        matrix_accumulator_92 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_135), 0x44004400u);

        matrix_accumulator_106 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101),
            select_half_operand<2, false, false>(packed_half_10), 0x3b283b28u);

        packed_half_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                select_half_operand<2, false, false>(matrix_accumulator_96));

        matrix_accumulator_94 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_139), 0x44004400u);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            uint64_t(uint32_t(matrix_fragment_99)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_158, address_word_159));

        matrix_accumulator_97 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97), 0xc400c400u);

        matrix_accumulator_92 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0xc400c400u);

        matrix_accumulator_94 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0xc400c400u);

        matrix_accumulator_93 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_162),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        split_address_words(matrix_accumulator_100, matrix_accumulator_101,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_100, matrix_accumulator_101));

        matrix_accumulator_107 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_97),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        packed_half_161 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_92),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        packed_half_160 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_94),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_84, matrix_fragment_85, matrix_accumulator_126,
                                 matrix_accumulator_127);

        matrix_accumulator_106 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_accumulator_107 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_97),
            select_half_operand<2, false, false>(matrix_accumulator_107), 0x3b283b28u);

        packed_half_162 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_162),
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6400ull,
            matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99,
            memory_ranges, error_counts);

        packed_half_161 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92),
            select_half_operand<2, false, false>(packed_half_161), 0x3b283b28u);

        packed_half_160 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94),
            select_half_operand<2, false, false>(packed_half_160), 0x3b283b28u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6200ull,
            matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_94,
            matrix_accumulator_95, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_128,
                                 matrix_fragment_129);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6000ull,
            matrix_fragment_88, matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
            memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_100, matrix_accumulator_101) + 0x6600ull,
            matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_102,
            matrix_fragment_103, memory_ranges, error_counts);

        packed_half_162 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                                select_half_operand<2, false, false>(packed_half_162));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                          select_half_operand<2, false, false>(0u));

        packed_half_160 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                                select_half_operand<2, false, false>(packed_half_160));

        matrix_accumulator_107 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                select_half_operand<2, false, false>(matrix_accumulator_107));

        matrix_fragment_105 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                          select_half_operand<2, false, false>(matrix_accumulator_106));

        packed_half_161 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                select_half_operand<2, false, false>(packed_half_161));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_162),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_135 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                               select_half_operand<2, false, false>(0u));

        packed_half_160 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_160),
                                        select_half_operand<2, false, false>(0u));

        matrix_accumulator_106 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                          select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_fragment_104 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                          select_half_operand<2, false, false>(matrix_accumulator_135));

        matrix_accumulator_107 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_161),
                          select_half_operand<2, false, false>(packed_half_160));

        multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_accumulator_80,
                                 matrix_accumulator_81, matrix_accumulator_120,
                                 matrix_accumulator_121);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_accumulator_82,
                                 matrix_accumulator_83, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_76, matrix_fragment_77, matrix_accumulator_80,
                                 matrix_accumulator_81);

        multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_78, matrix_fragment_79, matrix_accumulator_82,
                                 matrix_accumulator_83);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_84, matrix_fragment_85,
                                 matrix_accumulator_114, matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        packed_half_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_80), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_64, matrix_fragment_65, 0u, 0u);

        packed_half_10 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                                 matrix_fragment_53);

        multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_66, matrix_fragment_67, 0u, 0u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_82), 0x44004400u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_81), 0x44004400u);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_10),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_78, matrix_fragment_79, matrix_fragment_54,
                                 matrix_fragment_55);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_43),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_10),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        packed_half_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_83), 0x44004400u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_84,
                                 matrix_fragment_85);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        packed_half_10 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0xc400c400u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_72,
                                 matrix_fragment_73, matrix_fragment_74, matrix_fragment_75,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_48,
                                 matrix_fragment_49);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_10),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_10),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        packed_half_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_60, matrix_fragment_61, matrix_fragment_64,
                                 matrix_fragment_65);

        packed_half_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(packed_half_10), 0x3b283b28u);

        matrix_fragment_52 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(packed_half_10));

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_fragment_61 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

        matrix_fragment_60 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_68,
                                 matrix_fragment_69, matrix_fragment_70, matrix_fragment_71,
                                 matrix_fragment_62, matrix_fragment_63, matrix_fragment_40,
                                 matrix_fragment_41);

        packed_half_10 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_84), 0x44004400u);

        matrix_accumulator_80 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_67 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61), 0xc400c400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_60),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        packed_half_10 =
            maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0xc400c400u);

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

        matrix_fragment_41 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_85), 0x44004400u);

        matrix_accumulator_81 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

        matrix_fragment_61 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60),
            select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

        matrix_fragment_50 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(packed_half_10),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_41 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

        matrix_fragment_60 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_53 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_61 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_60), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(packed_half_10),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        matrix_accumulator_82 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_50),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        packed_half_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_41),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_43 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        packed_half_10 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41),
            select_half_operand<2, false, false>(packed_half_10), 0x3b283b28u);

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

        matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                          select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_49 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_61),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_72 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117,
                                 matrix_fragment_104, matrix_fragment_105, matrix_accumulator_106,
                                 matrix_accumulator_107, matrix_fragment_86, matrix_fragment_87,
                                 matrix_accumulator_116, matrix_accumulator_117);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

        matrix_fragment_49 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

        matrix_fragment_66 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_61),
            select_half_operand<2, false, false>(matrix_fragment_66), 0x3b283b28u);

        matrix_fragment_73 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_72), 0xc400c400u);

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_67),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_48 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_49),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_73),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_72 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_67),
            select_half_operand<2, false, false>(matrix_fragment_72), 0x3b283b28u);

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_74 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_73),
            select_half_operand<2, false, false>(matrix_fragment_74), 0x3b283b28u);

        packed_half_149 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_48),
            select_half_operand<0, false, true>(packed_half_149), 0x37283728u);

        matrix_fragment_60 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_60), 0x3b283b28u);

        matrix_fragment_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                select_half_operand<2, false, false>(matrix_fragment_72));

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        packed_half_10 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_85),
                                select_half_operand<2, false, false>(packed_half_10));

        matrix_fragment_50 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                select_half_operand<2, false, false>(matrix_fragment_74));

        packed_half_149 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48),
            select_half_operand<2, false, false>(packed_half_149), 0x3b283b28u);

        matrix_fragment_60 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                select_half_operand<2, false, false>(matrix_fragment_60));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                          select_half_operand<2, false, false>(matrix_fragment_61));

        matrix_fragment_66 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                select_half_operand<2, false, false>(matrix_fragment_66));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        packed_half_149 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                select_half_operand<2, false, false>(packed_half_149));

        matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                           select_half_operand<2, false, false>(0u));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                           select_half_operand<2, false, false>(packed_half_10));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                          select_half_operand<2, false, false>(matrix_fragment_60));

        multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_132,
                                 matrix_accumulator_133);

        multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_46, matrix_fragment_47, matrix_fragment_128,
                                 matrix_fragment_129);

        multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_126,
                                 matrix_accumulator_127);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_58, matrix_fragment_59, matrix_accumulator_116,
                                 matrix_accumulator_117);

        multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_46, matrix_fragment_47, matrix_accumulator_124,
                                 matrix_accumulator_125);

        multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_136,
                                 matrix_accumulator_137);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_56, matrix_fragment_57, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_44, matrix_fragment_45, matrix_accumulator_120,
                                 matrix_accumulator_121);

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                          select_half_operand<2, false, false>(0u));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_128),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                          select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_129),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                          select_half_operand<2, false, false>(packed_half_10));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_116),
                          select_half_operand<2, false, false>(0u));

        packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                          select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_44 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_114),
                          select_half_operand<2, false, false>(matrix_fragment_45));

        matrix_fragment_45 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                          select_half_operand<2, false, false>(packed_half_10));

        matrix_fragment_46 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_47 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_110,
                                 matrix_accumulator_111);

        uniform_condition_3 = uniform_condition_0;

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_88, matrix_fragment_89, matrix_accumulator_108,
                                 matrix_accumulator_109);

        multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_12,
                                 matrix_fragment_13);

        multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_90, matrix_fragment_91, matrix_fragment_14,
                                 matrix_fragment_15);

        multiply_accumulate_half(packed_half_16, matrix_accumulator_17, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_92, matrix_accumulator_93, packed_half_16,
                                 matrix_accumulator_17);

        multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_92, matrix_accumulator_93,
                                 matrix_accumulator_18, matrix_accumulator_19);

        multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_94, matrix_accumulator_95,
                                 matrix_accumulator_20, matrix_accumulator_21);

        multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_94, matrix_accumulator_95,
                                 matrix_accumulator_22, matrix_accumulator_23);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_98, matrix_fragment_99, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_98, matrix_fragment_99, matrix_fragment_30,
                                 matrix_fragment_31);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_32,
                                 matrix_fragment_33);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_102, matrix_fragment_103, matrix_fragment_34,
                                 matrix_fragment_35);

        multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_36,
                                 matrix_fragment_37);

        multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_102, matrix_fragment_103, matrix_fragment_38,
                                 matrix_fragment_39);

        packed_half_10 = 0x1u;

        uniform_condition_0 = false;
    } while (uniform_condition_3);
    packed_half_161 = threadIdx.y;

    uniform_index_word_8 = 0u;

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_4 = 0x400u;

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(address_word_2));

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                      select_half_operand<2, false, false>(address_word_2));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    uniform_index_word_4 = (uniform_index_word_8 << 0x18u) + uniform_index_word_4;

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                   select_half_operand<2, false, false>(0u));

    uniform_index_word_8 = parameters[parameter_scalar_control_11];

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = packed_half_161 * 0x800u;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(address_word_2));

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 = packed_half_3 + uniform_index_word_4;

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                                   select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                      select_half_operand<2, false, false>(address_word_2));

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = address_word_157 * 0x10u + matrix_accumulator_20;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                   select_half_operand<2, false, false>(0u));

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                   select_half_operand<2, false, false>(0u));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_28, matrix_fragment_4,
                                         matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                         shared_bytes, memory_ranges, error_counts);

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                      select_half_operand<2, false, false>(matrix_accumulator_23));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_28 + 0x200u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_17 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = packed_half_161 * 0x300u;

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                          select_half_operand<2, false, false>(packed_half_11));

    matrix_accumulator_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(address_word_2));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_28 + 0x400u, packed_half_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                          select_half_operand<2, false, false>(packed_half_10));

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_fragment_28 + 0x600u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_accumulator_22, matrix_accumulator_23, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x72a0ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x70a0ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x74a0ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x76a0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x78a0ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x7aa0ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x8aa0ull, matrix_fragment_68,
        matrix_fragment_69, matrix_fragment_70, matrix_fragment_71, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x88a0ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x8ca0ull, matrix_fragment_140,
        matrix_fragment_141, matrix_fragment_142, matrix_fragment_143, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x90a0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x8ea0ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x92a0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    address_word_2 = (address_word_157 << 0x4u) + uniform_index_word_4;

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0x400u, matrix_fragment_8,
                                        matrix_fragment_9, packed_half_10, packed_half_11,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, address_word_2 + 0x200u, matrix_fragment_128, matrix_fragment_129,
        matrix_fragment_130, matrix_fragment_131, shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0x600u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0x800u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0xc00u, matrix_fragment_60,
                                        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                                        shared_bytes, memory_ranges, error_counts);

    split_address_words(packed_half_154, packed_half_155,
                        uint64_t(uint32_t(packed_half_161)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    packed_half_154 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_154, packed_half_155) + 0xe0a0ull, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_150, matrix_fragment_151, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_76, matrix_fragment_77, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_78, matrix_fragment_79, 0u, 0u);

    multiply_accumulate_half(packed_half_16, matrix_accumulator_17, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_74,
                             matrix_fragment_75, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_fragment_150, matrix_fragment_151, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_150,
                             matrix_fragment_151);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_104,
                             matrix_fragment_105);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_26,
                             matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_32,
                             matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_34,
                             matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_36,
                             matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_38,
                             matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_40,
                             matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_42,
                             matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_24,
                             matrix_fragment_25, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_72,
                             matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_76,
                             matrix_fragment_77, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_8,
                             matrix_fragment_9, packed_half_10, packed_half_11, matrix_fragment_78,
                             matrix_fragment_79, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_142, matrix_fragment_143, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_122, matrix_accumulator_123, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_122,
                             matrix_accumulator_123);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_58, matrix_fragment_59, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_140, matrix_fragment_141, matrix_fragment_90,
                             matrix_fragment_91);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_64,
                             matrix_fragment_65);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_fragment_128, matrix_fragment_129, matrix_fragment_128,
                             matrix_fragment_129, matrix_fragment_130, matrix_fragment_131,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_14,
                             matrix_fragment_15);

    multiply_accumulate_half(matrix_fragment_130, matrix_fragment_131, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_58, matrix_fragment_59, packed_half_16,
                             matrix_accumulator_17);

    packed_half_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_150),
                            select_half_operand<2, false, false>(matrix_fragment_150));

    packed_half_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(packed_half_16));

    matrix_accumulator_17 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(matrix_accumulator_17));

    multiply_accumulate_half(packed_half_10, packed_half_11, matrix_fragment_44, matrix_fragment_45,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_6,
                             matrix_fragment_7, matrix_accumulator_22, matrix_accumulator_23);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_40 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                                        select_half_operand<2, false, false>(packed_half_16));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    packed_half_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_151),
                            select_half_operand<2, false, false>(matrix_fragment_151));

    packed_half_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(packed_half_16));

    matrix_accumulator_17 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_107),
                                      select_half_operand<2, false, false>(matrix_accumulator_17));

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_120, matrix_accumulator_121, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_120,
                             matrix_accumulator_121);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_140, matrix_fragment_141, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_142, matrix_fragment_143, matrix_accumulator_80,
                             matrix_accumulator_81);

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_18,
                             matrix_accumulator_19);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_132,
                             matrix_accumulator_133);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_118, matrix_accumulator_119, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_118,
                             matrix_accumulator_119);

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_44 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                                        select_half_operand<2, false, false>(packed_half_16));

    split_address_words(packed_half_16, matrix_accumulator_17,
                        uint64_t(uint32_t(packed_half_3)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(matrix_fragment_45));

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_46));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_44), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_3 = result_word;
    }

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_25), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_96 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    matrix_accumulator_94 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                       select_half_operand<2, false, false>(matrix_accumulator_94));

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0xe00u, matrix_fragment_64,
                                        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_84, matrix_fragment_85, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0xa00u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                   select_half_operand<2, false, false>(packed_half_3));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_94), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_97 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_25 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_fragment_27 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                             select_half_operand<2, false, false>(packed_half_10));

    matrix_fragment_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_3), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_26 = result_word;
    }

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_44 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                             select_half_operand<2, false, false>(packed_half_11));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_25), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_98 = result_word;
    }

    matrix_fragment_24 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_46 = permute_signed(matrix_fragment_45, matrix_fragment_45, 0x5432u);

    matrix_fragment_27 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_44 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_44));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_24), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    matrix_accumulator_95 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                       select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_27 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_94 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                       select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_27), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_96 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_94), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_95 = result_word;
    }

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_26));

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_140, matrix_fragment_141, matrix_accumulator_22,
                             matrix_accumulator_23);

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_98));

    multiply_accumulate_half(matrix_fragment_140, matrix_fragment_141, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_140, matrix_fragment_141, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_fragment_87 = permute_signed(matrix_fragment_26, matrix_fragment_26, 0x5432u);

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_87));

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_24 = permute_signed(matrix_fragment_25, matrix_fragment_25, 0x5432u);

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    packed_half_3 = half_to_float_bits((matrix_accumulator_97 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_100 = result_word;
    }

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_86 = half_to_float_bits((matrix_accumulator_97 >> 16) * 65537u);

    matrix_accumulator_97 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x4100410u);

    split_address_words(packed_half_16, matrix_accumulator_17,
                        int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_16, matrix_accumulator_17));

    __syncthreads();

    matrix_accumulator_94 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_95 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x4100410u);

    matrix_fragment_98 = half_to_float_bits((matrix_accumulator_97 & 65535u) * 65537u);

    matrix_accumulator_97 = half_to_float_bits((matrix_accumulator_97 >> 16) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_96), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_99 = result_word;
    }

    packed_half_3 = rsqrt_bits(packed_half_3);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_142, matrix_fragment_143, matrix_accumulator_124,
                             matrix_accumulator_125);

    matrix_fragment_86 = rsqrt_bits(matrix_fragment_86);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xa2a0ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    matrix_accumulator_101 = rsqrt_bits(matrix_accumulator_97);

    multiply_accumulate_half(matrix_fragment_142, matrix_fragment_143, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_142, matrix_fragment_143, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_87 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xa0a0ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xa4a0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    matrix_accumulator_97 = permute_signed(matrix_accumulator_94, matrix_accumulator_94, 0x5432u);

    matrix_fragment_85 = half_to_float_bits((matrix_accumulator_95 & 65535u) * 65537u);

    matrix_accumulator_95 = half_to_float_bits((matrix_accumulator_95 >> 16) * 65537u);

    matrix_fragment_84 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                       select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_fragment_98 = rsqrt_bits(matrix_fragment_98);

    matrix_fragment_84 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_84), 0x4100410u);

    matrix_accumulator_94 = rsqrt_bits(matrix_fragment_85);

    matrix_fragment_102 = half_to_float_bits((matrix_fragment_84 & 65535u) * 65537u);

    packed_half_153 = pack_half(matrix_fragment_86, packed_half_3);

    matrix_accumulator_100 = half_to_float_bits((matrix_fragment_84 >> 16) * 65537u);

    matrix_fragment_84 = permute_signed(matrix_fragment_87, matrix_fragment_87, 0x5432u);

    matrix_accumulator_95 = rsqrt_bits(matrix_accumulator_95);

    matrix_fragment_99 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                        select_half_operand<2, false, false>(matrix_fragment_99));

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(packed_half_153));

    packed_half_3 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                   select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_92 = permute_signed(matrix_fragment_99, matrix_fragment_99, 0x5432u);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_134,
                             matrix_accumulator_135);

    matrix_fragment_98 = pack_half(matrix_accumulator_101, matrix_fragment_98);

    matrix_accumulator_97 = rsqrt_bits(matrix_fragment_102);

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x4100410u);

    matrix_fragment_99 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99),
                       select_half_operand<2, false, false>(matrix_accumulator_92));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xa6a0ull, matrix_fragment_84,
        matrix_fragment_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x4100410u);

    matrix_accumulator_94 = pack_half(matrix_accumulator_95, matrix_accumulator_94);

    matrix_accumulator_92 = half_to_float_bits((packed_half_3 >> 16) * 65537u);

    multiply_accumulate_half(matrix_accumulator_144, matrix_accumulator_145, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_4, matrix_fragment_5, matrix_accumulator_144,
                             matrix_accumulator_145);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(packed_half_153));

    matrix_accumulator_96 = rsqrt_bits(matrix_accumulator_100);

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_150),
                                          select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_4 = half_to_float_bits((packed_half_3 & 65535u) * 65537u);

    packed_half_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_fragment_103 = half_to_float_bits((matrix_fragment_99 >> 16) * 65537u);

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_99 & 65535u) * 65537u);

    matrix_fragment_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    packed_half_113 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_accumulator_93 = rsqrt_bits(matrix_accumulator_92);

    multiply_accumulate_half(matrix_accumulator_146, matrix_accumulator_147, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_146,
                             matrix_accumulator_147);

    matrix_fragment_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_accumulator_120),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_3 = rsqrt_bits(matrix_fragment_88);

    packed_half_160 = rsqrt_bits(matrix_fragment_103);

    packed_half_113 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                     select_half_operand<2, false, false>(packed_half_113));

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_6, matrix_fragment_7, matrix_accumulator_108,
                             matrix_accumulator_109);

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_accumulator_96 = pack_half(matrix_accumulator_96, matrix_accumulator_97);

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_accumulator_121),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = pack_half(matrix_accumulator_93, matrix_fragment_4);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 = pack_half(packed_half_160, packed_half_3);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    packed_half_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_accumulator_96));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xaaa0ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    matrix_accumulator_148 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xa8a0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                            select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    packed_half_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_accumulator_134));

    matrix_accumulator_18 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(matrix_accumulator_22),
                                      select_half_operand<2, false, false>(packed_half_11));

    packed_half_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(matrix_accumulator_146));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    packed_half_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_accumulator_19 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(matrix_accumulator_124),
                                      select_half_operand<2, false, false>(packed_half_11));

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_151 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_151),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_96));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xaca0ull, matrix_accumulator_96,
        matrix_accumulator_97, matrix_fragment_98, matrix_fragment_99, memory_ranges, error_counts);

    packed_half_11 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(matrix_accumulator_23),
                                      select_half_operand<2, false, false>(packed_half_10));

    packed_half_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                                    select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(matrix_accumulator_101));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xaea0ull, matrix_accumulator_100,
        matrix_accumulator_101, matrix_fragment_102, matrix_fragment_103, memory_ranges,
        error_counts);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(matrix_accumulator_147));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_10), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_155 = result_word;
    }

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_accumulator_125),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                    select_half_operand<2, false, false>(packed_half_11));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_113), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_149 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_11), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    packed_half_163 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                      select_half_operand<2, false, false>(matrix_accumulator_19));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                     select_half_operand<2, false, false>(packed_half_155));

    packed_half_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(matrix_accumulator_144));

    matrix_accumulator_148 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_148),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    packed_half_112 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_140),
                                      select_half_operand<2, false, false>(matrix_fragment_140),
                                      select_half_operand<2, false, false>(packed_half_10));

    packed_half_113 = add_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                     select_half_operand<2, false, false>(matrix_fragment_15));

    packed_half_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_accumulator_145));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(matrix_accumulator_148));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_142),
                                      select_half_operand<2, false, false>(matrix_fragment_142),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    packed_half_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_141),
                                      select_half_operand<2, false, false>(matrix_fragment_141),
                                      select_half_operand<2, false, false>(packed_half_10));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(packed_half_149));

    matrix_accumulator_148 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_143),
                                      select_half_operand<2, false, false>(matrix_fragment_143),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_155), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_162 = result_word;
    }

    packed_half_11 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                    select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_112));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_149 = result_word;
    }

    matrix_fragment_12 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_148),
                       select_half_operand<2, false, false>(packed_half_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_113), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_11), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_164 = result_word;
    }

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                      select_half_operand<2, false, false>(packed_half_163));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_163 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_160 = result_word;
    }

    packed_half_162 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(packed_half_149));

    multiply_accumulate_half(matrix_accumulator_148, packed_half_149, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    packed_half_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(packed_half_10));

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                     select_half_operand<2, false, false>(packed_half_164));

    packed_half_11 = pack_half(0u, packed_half_154);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                          select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_73 = permute_signed(packed_half_162, packed_half_162, 0x5432u);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_162 = add_half_pairs(select_half_operand<2, false, false>(packed_half_162),
                                     select_half_operand<2, false, false>(matrix_fragment_73));

    packed_half_165 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                    select_half_operand<2, false, false>(0u));

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                          select_half_operand<0, false, false>(packed_half_11));

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_162), 0x4100410u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_163));

    packed_half_163 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                           select_half_operand<2, false, false>(packed_half_165));

    packed_half_152 = half_to_float_bits((packed_half_153 & 65535u) * 65537u);

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                        select_half_operand<2, false, false>(packed_half_160));

    packed_half_160 = half_to_float_bits((packed_half_153 >> 16) * 65537u);

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_151),
                                          select_half_operand<0, false, false>(packed_half_11));

    packed_half_154 = permute_signed(packed_half_155, packed_half_155, 0x5432u);

    packed_half_152 = rsqrt_bits(packed_half_152);

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_151 = rsqrt_bits(packed_half_160);

    packed_half_155 = add_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                                     select_half_operand<2, false, false>(packed_half_154));

    packed_half_154 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                           select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_153 = result_word;
    }

    matrix_fragment_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_150),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_151 = pack_half(matrix_fragment_151, packed_half_152);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_152 = result_word;
    }

    packed_half_155 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_155), 0x4100410u);

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_150),
                      select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_150 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                        select_half_operand<2, false, false>(packed_half_154));

    packed_half_154 = half_to_float_bits((packed_half_155 & 65535u) * 65537u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(0u));

    packed_half_155 = half_to_float_bits((packed_half_155 >> 16) * 65537u);

    matrix_fragment_9 = permute_signed(packed_half_10, packed_half_10, 0x5432u);

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                     select_half_operand<2, false, false>(packed_half_153));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(0u));

    packed_half_154 = rsqrt_bits(packed_half_154);

    packed_half_10 = add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                    select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 = permute_signed(packed_half_153, packed_half_153, 0x5432u);

    packed_half_155 = rsqrt_bits(packed_half_155);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_152));

    packed_half_153 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                     select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_10 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_10), 0x4100410u);

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_151));

    matrix_fragment_5 = permute_signed(matrix_fragment_12, matrix_fragment_12, 0x5432u);

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_124),
                            select_half_operand<2, false, false>(matrix_fragment_151));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_151));

    matrix_accumulator_146 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_146),
                            select_half_operand<2, false, false>(matrix_fragment_151));

    matrix_fragment_151 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                        select_half_operand<2, false, false>(packed_half_163));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    packed_half_153 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_153), 0x4100410u);

    packed_half_3 = half_to_float_bits((packed_half_10 & 65535u) * 65537u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 = half_to_float_bits((packed_half_10 >> 16) * 65537u);

    matrix_fragment_5 = half_to_float_bits((packed_half_153 & 65535u) * 65537u);

    packed_half_10 = half_to_float_bits((packed_half_153 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_150, matrix_fragment_151, matrix_fragment_44,
                             matrix_fragment_45);

    packed_half_154 = pack_half(packed_half_155, packed_half_154);

    matrix_fragment_5 = rsqrt_bits(matrix_fragment_5);

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_125),
                            select_half_operand<2, false, false>(packed_half_154));

    packed_half_10 = rsqrt_bits(packed_half_10);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_47 = permute_signed(matrix_fragment_6, matrix_fragment_6, 0x5432u);

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_22 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7), 0x4100410u);

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22), 0x4100410u);

    matrix_accumulator_124 = half_to_float_bits((matrix_fragment_7 & 65535u) * 65537u);

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_7 = half_to_float_bits((matrix_fragment_7 >> 16) * 65537u);

    matrix_accumulator_23 = half_to_float_bits((matrix_accumulator_22 & 65535u) * 65537u);

    matrix_accumulator_22 = half_to_float_bits((matrix_accumulator_22 >> 16) * 65537u);

    packed_half_3 = rsqrt_bits(packed_half_3);

    packed_half_10 = pack_half(packed_half_10, matrix_fragment_5);

    matrix_fragment_4 = rsqrt_bits(matrix_fragment_4);

    matrix_fragment_6 = rsqrt_bits(matrix_accumulator_124);

    matrix_fragment_7 = rsqrt_bits(matrix_fragment_7);

    matrix_fragment_5 = rsqrt_bits(matrix_accumulator_23);

    matrix_accumulator_22 = rsqrt_bits(matrix_accumulator_22);

    matrix_fragment_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_142),
                            select_half_operand<2, false, false>(packed_half_10));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(packed_half_10));

    matrix_accumulator_147 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_147),
                            select_half_operand<2, false, false>(packed_half_154));

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_135 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_142),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_140),
                            select_half_operand<2, false, false>(packed_half_10));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_144),
                            select_half_operand<2, false, false>(packed_half_10));

    packed_half_3 = pack_half(matrix_fragment_4, packed_half_3);

    matrix_fragment_7 = pack_half(matrix_fragment_7, matrix_fragment_6);

    matrix_accumulator_124 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                      select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_5 = pack_half(matrix_accumulator_22, matrix_fragment_5);

    matrix_accumulator_125 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                      select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_146),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_143 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_143),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_147),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(packed_half_11));

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_84,
                             matrix_fragment_85);

    matrix_fragment_141 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_141),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_145 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_145),
                            select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_143),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_120),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_121),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<0, false, false>(packed_half_11));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_141),
                                       select_half_operand<2, false, false>(matrix_fragment_84));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_145),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<0, false, false>(packed_half_11));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_120),
                                  select_half_operand<2, false, false>(0u));

    packed_half_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_121),
                                   select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(packed_half_10));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                      select_half_operand<2, false, false>(matrix_accumulator_22));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_24,
                             matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_26,
                             matrix_fragment_27);

    packed_half_3 = 0x29c0u;

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_104,
                             matrix_fragment_105, matrix_accumulator_106, matrix_accumulator_107,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_accumulator_80 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_88,
                             matrix_fragment_89);

    matrix_accumulator_82 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_150, matrix_fragment_151, matrix_accumulator_92,
                             matrix_accumulator_93);

    matrix_accumulator_81 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_94,
                             matrix_accumulator_95);

    matrix_accumulator_83 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_15),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_13),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_80 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3c203c20u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_82 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3c203c20u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3c203c20u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_44, matrix_fragment_45, matrix_accumulator_96,
                             matrix_accumulator_97);

    matrix_accumulator_81 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3c203c20u);

    matrix_accumulator_83 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3c203c20u);

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3c203c20u);

    matrix_fragment_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3c203c20u);

    matrix_accumulator_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_80 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_80), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_accumulator_82 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_82), 0x3e473e47u);

    matrix_fragment_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_86),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_14 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3e473e47u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3c203c20u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3c203c20u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3c203c20u);

    matrix_accumulator_81 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_81), 0x3e473e47u);

    matrix_accumulator_83 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_83), 0x3e473e47u);

    packed_half_10 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3e473e47u);

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_102, matrix_fragment_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_102,
                             matrix_fragment_103);

    matrix_fragment_104 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3e473e47u);

    matrix_fragment_88 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_88),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3e473e47u);

    matrix_fragment_89 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_89),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3e473e47u);

    matrix_fragment_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_90),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3e473e47u);

    matrix_fragment_91 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_91),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_15 = (matrix_accumulator_80 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_23 = (matrix_accumulator_82 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_25 = (matrix_fragment_14 << 0x5u) + 0x7ff88000u;

    matrix_fragment_27 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_22 = (matrix_accumulator_81 << 0x5u) + 0x7ff88000u;

    matrix_fragment_24 = (matrix_accumulator_83 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 = (packed_half_10 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_80 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_108 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108), 0x3c203c20u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3c203c20u);

    matrix_accumulator_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_10 = (matrix_fragment_104 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_13 = (matrix_accumulator_106 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_12 = (matrix_fragment_105 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_82 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_14 = (matrix_accumulator_107 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                       select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_accumulator_108 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108), 0x3e473e47u);

    matrix_fragment_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_86 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x3e473e47u);

    matrix_fragment_88 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3c203c20u);

    matrix_fragment_89 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3c203c20u);

    matrix_fragment_90 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3c203c20u);

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3c203c20u);

    matrix_accumulator_93 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3c203c20u);

    matrix_accumulator_95 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3c203c20u);

    matrix_accumulator_81 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_accumulator_92 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3c203c20u);

    matrix_fragment_105 = add_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                         select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_94 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x3c203c20u);

    matrix_fragment_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                       select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_88 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88), 0x3e473e47u);

    matrix_accumulator_82 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_89 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x3e473e47u);

    matrix_accumulator_93 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3e473e47u);

    matrix_accumulator_95 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3e473e47u);

    matrix_fragment_4 = (matrix_accumulator_108 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = (matrix_fragment_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_105 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                       select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_accumulator_92 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3e473e47u);

    matrix_accumulator_94 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x3e473e47u);

    matrix_fragment_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_6 = (matrix_fragment_88 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = (matrix_fragment_89 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_81 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                        select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_82 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_88 = (matrix_accumulator_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_98),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_89 = (matrix_accumulator_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_87),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_83 = (matrix_accumulator_92 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_86 = (matrix_accumulator_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_99),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_105 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                                         select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_96 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3c203c20u);

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_98 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3c203c20u);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_102),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3c203c20u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_103),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_97 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3c203c20u);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3c203c20u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                                        select_half_operand<2, false, false>(matrix_fragment_86));

    matrix_accumulator_109 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109), 0x3c203c20u);

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3e473e47u);

    matrix_fragment_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_98), 0x3e473e47u);

    matrix_fragment_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0x3c203c20u);

    matrix_fragment_87 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x3e473e47u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_fragment_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103), 0x3c203c20u);

    matrix_accumulator_97 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3e473e47u);

    matrix_fragment_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_99), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_76, matrix_fragment_77, 0u, 0u);

    matrix_accumulator_109 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_109), 0x3e473e47u);

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_accumulator_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0x3e473e47u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_accumulator_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103), 0x3e473e47u);

    matrix_fragment_90 = (matrix_fragment_90 << 0x5u) + 0x7ff88000u;

    matrix_fragment_103 = (matrix_fragment_87 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_76, matrix_fragment_77, 0u, 0u);

    matrix_fragment_87 = (matrix_accumulator_97 << 0x5u) + 0x7ff88000u;

    matrix_fragment_91 = (matrix_fragment_91 << 0x5u) + 0x7ff88000u;

    matrix_fragment_102 = (matrix_accumulator_109 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_92 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_100 = (matrix_accumulator_108 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_93 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_108 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_101 = (matrix_accumulator_110 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_97 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_accumulator_106 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                                        select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                       select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_96 = address_word_157 * 0x4u;

    condition_0 = (address_word_157 & 0x1u) != 0;

    matrix_accumulator_107 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                       select_half_operand<2, false, false>(matrix_accumulator_106));

    condition_1 = (address_word_157 & 0x2u) != 0;

    matrix_accumulator_108 = matrix_accumulator_96 & 0x1cu;

    matrix_accumulator_106 = condition_0 ? matrix_fragment_76 : matrix_fragment_105;

    matrix_accumulator_109 = condition_0 ? matrix_accumulator_107 : matrix_fragment_104;

    matrix_fragment_105 = condition_0 ? matrix_fragment_105 : matrix_fragment_76;

    matrix_fragment_104 = condition_0 ? matrix_fragment_104 : matrix_accumulator_107;

    matrix_fragment_76 = uint32_t(int64_t(int32_t(address_word_157)) * (1ll << 0x1du) >> 32) +
                         matrix_accumulator_108;

    matrix_accumulator_107 = !condition_1 ? matrix_accumulator_106 : matrix_accumulator_109;

    matrix_accumulator_106 = !condition_1 ? matrix_accumulator_109 : matrix_accumulator_106;

    matrix_accumulator_134 = 0x1u ^ matrix_fragment_76;

    matrix_accumulator_109 = !condition_1 ? matrix_fragment_105 : matrix_fragment_104;

    matrix_fragment_105 = !condition_1 ? matrix_fragment_104 : matrix_fragment_105;

    matrix_accumulator_121 = 0x2u ^ matrix_fragment_76;

    matrix_accumulator_120 = 0x3u ^ matrix_fragment_76;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_107), "r"(matrix_fragment_76), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_107 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_109), "r"(matrix_accumulator_134), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_104 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_106), "r"(matrix_accumulator_121), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_106 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_105), "r"(matrix_accumulator_120), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_105 = result_word;
    }

    condition_2 = (address_word_157 & 0x8u) != 0;

    condition_3 = (address_word_157 & 0x10u) != 0;

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_56, matrix_fragment_57, packed_half_112,
                             packed_half_113);

    matrix_accumulator_108 = !condition_2 ? matrix_accumulator_107 : matrix_fragment_104;

    matrix_fragment_104 = !condition_2 ? matrix_fragment_104 : matrix_accumulator_107;

    multiply_accumulate_half(matrix_accumulator_148, packed_half_149, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_56, matrix_fragment_57, matrix_accumulator_148,
                             packed_half_149);

    matrix_accumulator_111 = !condition_2 ? matrix_accumulator_106 : matrix_fragment_105;

    matrix_accumulator_107 = !condition_2 ? matrix_fragment_105 : matrix_accumulator_106;

    matrix_fragment_56 = !condition_3 ? matrix_accumulator_108 : matrix_accumulator_111;

    matrix_fragment_57 = !condition_3 ? matrix_fragment_104 : matrix_accumulator_107;

    matrix_accumulator_108 = !condition_3 ? matrix_accumulator_111 : matrix_accumulator_108;

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_104 = !condition_3 ? matrix_accumulator_107 : matrix_fragment_104;

    matrix_fragment_56 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                       select_half_operand<2, false, false>(matrix_accumulator_108));

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_74, matrix_fragment_75, 0u, 0u);

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_56 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_56),
                                        select_half_operand<1, false, false>(matrix_fragment_56));

    matrix_fragment_56 = permute_signed(matrix_fragment_56, matrix_fragment_56, 0x5410u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x4100410u);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_fragment_104 = half_to_float_bits((matrix_fragment_56 >> 16) * 65537u);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_58, matrix_fragment_59, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_104 = reciprocal_bits(matrix_fragment_104);

    matrix_fragment_59 = half_to_float_bits((matrix_fragment_56 & 65535u) * 65537u);

    matrix_fragment_56 = uint32_t(int32_t(index_word_0) >> 0x2u);

    matrix_fragment_59 = reciprocal_bits(matrix_fragment_59);

    matrix_fragment_57 = matrix_fragment_56 + 0x8u;

    matrix_fragment_105 = uint32_t(int32_t(index_word_0) >> 0x1fu);

    matrix_fragment_58 = uint32_t(int32_t(matrix_fragment_57) >> 0x1fu);

    matrix_fragment_105 =
        uint32_t(combine_words(matrix_fragment_105, 0u) << 0x5u >> 32) + matrix_fragment_56;

    matrix_accumulator_106 =
        uint32_t(combine_words(matrix_fragment_58, 0u) << 0x5u >> 32) + matrix_fragment_57;

    matrix_accumulator_106 = matrix_accumulator_106 & 0xffffffe0u;

    matrix_accumulator_136 = pack_half(matrix_fragment_104, matrix_fragment_59);

    matrix_fragment_59 = matrix_fragment_105 & 0xffffffe0u;

    matrix_accumulator_107 = matrix_fragment_57 + (0u - matrix_accumulator_106);

    matrix_fragment_59 = matrix_fragment_56 + (0u - matrix_fragment_59);

    matrix_fragment_131 = transpose_half_tile(matrix_fragment_131);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_78, matrix_fragment_79, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_136), "r"(matrix_accumulator_107), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_107 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_136), "r"(matrix_fragment_59), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_105 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_78, matrix_fragment_79, 0u, 0u);

    matrix_accumulator_115 = transpose_half_tile(matrix_accumulator_115);

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_131),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_131 = permute_signed(matrix_accumulator_107, matrix_accumulator_107, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_94,
                             matrix_accumulator_95);

    matrix_accumulator_123 = transpose_half_tile(matrix_accumulator_123);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_98,
                             matrix_fragment_99);

    matrix_fragment_52 = transpose_half_tile(matrix_accumulator_114);

    matrix_fragment_53 = transpose_half_tile(matrix_fragment_130);

    matrix_fragment_130 = permute_signed(matrix_fragment_105, matrix_fragment_105, 0x5410u);

    matrix_accumulator_122 = transpose_half_tile(matrix_accumulator_122);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_96,
                             matrix_accumulator_97);

    matrix_accumulator_23 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_23),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_accumulator_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_22),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_80),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    matrix_accumulator_110 =
        uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_15 = uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_accumulator_22 = uint32_t(int32_t(matrix_fragment_15) >> 0x1fu);

    matrix_fragment_105 = uint32_t(int32_t(matrix_accumulator_110) >> 0x1fu);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 =
        uint32_t(combine_words(matrix_accumulator_22, 0u) << 0x5u >> 32) + matrix_fragment_15;

    matrix_fragment_105 =
        uint32_t(combine_words(matrix_fragment_105, 0u) << 0x5u >> 32) + matrix_accumulator_110;

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_accumulator_80 = matrix_accumulator_80 & 0xffffffe0u;

    matrix_accumulator_123 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_123),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_111 = matrix_fragment_105 & 0xffffffe0u;

    matrix_accumulator_117 = transpose_half_tile(matrix_accumulator_117);

    matrix_accumulator_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_22 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_122),
                      select_half_operand<2, false, false>(matrix_accumulator_123));

    matrix_fragment_15 = matrix_fragment_15 + (0u - matrix_accumulator_80);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_accumulator_123 = matrix_accumulator_110 + (0u - matrix_accumulator_111);

    matrix_fragment_55 = transpose_half_tile(matrix_accumulator_116);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_accumulator_133 = transpose_half_tile(matrix_accumulator_133);

    matrix_accumulator_127 = transpose_half_tile(matrix_accumulator_127);

    matrix_fragment_129 = transpose_half_tile(matrix_fragment_129);

    matrix_accumulator_119 = transpose_half_tile(matrix_accumulator_119);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_136), "r"(matrix_fragment_15), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_accumulator_132 = transpose_half_tile(matrix_accumulator_132);

    matrix_fragment_54 = transpose_half_tile(matrix_accumulator_126);

    matrix_fragment_104 = transpose_half_tile(matrix_fragment_128);

    matrix_fragment_105 = transpose_half_tile(matrix_accumulator_118);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_136), "r"(matrix_accumulator_123), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_123 = result_word;
    }

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_129 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_129),
                                        select_half_operand<2, false, false>(0u));

    matrix_accumulator_80 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = permute_signed(matrix_fragment_15, matrix_fragment_15, 0x5410u);

    matrix_accumulator_23 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_fragment_54 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                        select_half_operand<2, false, false>(matrix_fragment_129));

    matrix_fragment_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                      select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_accumulator_116 =
        permute_signed(matrix_accumulator_123, matrix_accumulator_123, 0x5410u);

    matrix_fragment_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_accumulator_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_82),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_81),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_81 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_83),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_accumulator_82));

    matrix_fragment_7 = transpose_half_tile(matrix_fragment_73);

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_accumulator_81));

    matrix_fragment_6 = transpose_half_tile(matrix_fragment_72);

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    matrix_fragment_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    matrix_fragment_75 = transpose_half_tile(matrix_fragment_75);

    matrix_fragment_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_103),
                            select_half_operand<2, false, false>(matrix_fragment_131));

    packed_half_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_10),
                                         select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_fragment_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_fragment_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_130));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_97 = transpose_half_tile(matrix_accumulator_97);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_131),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = transpose_half_tile(packed_half_113);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_102 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                        select_half_operand<2, false, false>(matrix_fragment_7));

    packed_half_149 = transpose_half_tile(packed_half_149);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_accumulator_95 = transpose_half_tile(matrix_accumulator_95);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_12 = transpose_half_tile(matrix_fragment_99);

    matrix_fragment_14 = transpose_half_tile(matrix_fragment_79);

    matrix_accumulator_96 = transpose_half_tile(matrix_accumulator_96);

    matrix_fragment_86 = transpose_half_tile(matrix_fragment_74);

    packed_half_112 = transpose_half_tile(packed_half_112);

    matrix_accumulator_148 = transpose_half_tile(matrix_accumulator_148);

    matrix_accumulator_94 = transpose_half_tile(matrix_accumulator_94);

    packed_half_10 = transpose_half_tile(matrix_fragment_98);

    matrix_fragment_13 = transpose_half_tile(matrix_fragment_78);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_accumulator_80));

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                        select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    packed_half_149 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_accumulator_96 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                      select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_fragment_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_86),
                                        select_half_operand<2, false, false>(matrix_fragment_103));

    matrix_fragment_98 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_148),
                                       select_half_operand<2, false, false>(packed_half_149));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_accumulator_95));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_accumulator_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_130),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_10),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_108,
                             matrix_accumulator_109);

    packed_half_10 = packed_half_161 * 0x20u;

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_fragment_4 = index_word_0 & 0x7ffffffcu;

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_102, matrix_fragment_103, matrix_accumulator_82,
                             matrix_accumulator_83);

    matrix_fragment_7 = 0u - matrix_fragment_4 + address_word_157;

    matrix_fragment_5 = uint32_t(uint32_t(packed_half_10) >> 0x1u);

    matrix_fragment_7 = matrix_fragment_7 * 0x2u + packed_half_10;

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_74,
                             matrix_fragment_75);

    matrix_fragment_4 = matrix_fragment_5 + 0x4u;

    matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_7) >> 0x1u);

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_80,
                             matrix_accumulator_81);

    matrix_fragment_6 = matrix_fragment_5 + 0x8u;

    split_address_words(matrix_accumulator_114, matrix_accumulator_115,
                        int64_t(int32_t(matrix_fragment_7)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_114 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_114, matrix_accumulator_115) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_15 = matrix_fragment_4 + address_word_157 + (0u - index_word_156);

    matrix_fragment_73 = matrix_fragment_6 + address_word_157 + (0u - index_word_156);

    matrix_fragment_5 = matrix_fragment_5 + 0xcu;

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        uint64_t(uint32_t(matrix_fragment_15)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = (packed_half_161 << 0x9u) + uniform_index_word_4;

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        uint64_t(uint32_t(matrix_fragment_73)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_87 = matrix_fragment_5 + address_word_157 + (0u - index_word_156);

    matrix_accumulator_115 = address_word_157 * 0x10u + matrix_fragment_7;

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_77 = packed_half_161 * 0x100u;

    packed_half_112 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_72, matrix_fragment_73) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_88));

    split_address_words(matrix_fragment_86, matrix_fragment_87,
                        uint64_t(uint32_t(matrix_fragment_87)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_74 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_accumulator_83));

    matrix_fragment_75 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_82));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_115, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    split_address_words(address_word_158, address_word_159,
                        uint64_t(uint32_t(matrix_fragment_77)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_158, address_word_159));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_115, matrix_fragment_24,
                                         matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                                         shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_115 + 0x400u, matrix_fragment_24, matrix_fragment_25,
        matrix_fragment_26, matrix_fragment_27, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_77 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xf0b0ull, memory_ranges,
        error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_115 + 0x400u, matrix_fragment_72, matrix_fragment_73,
        matrix_fragment_74, matrix_fragment_75, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_86, matrix_fragment_87,
                        int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_158, address_word_159));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xe0b0ull, matrix_fragment_72,
        matrix_fragment_73, matrix_fragment_74, matrix_fragment_75, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xe2b0ull, matrix_accumulator_80,
        matrix_accumulator_81, matrix_accumulator_82, matrix_accumulator_83, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xe8b0ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xeab0ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_fragment_32 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_13);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_106,
                             matrix_accumulator_107);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_70, matrix_fragment_71, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_34,
                             matrix_fragment_35);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(packed_half_113));

    multiply_accumulate_half(matrix_fragment_70, matrix_fragment_71, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_70, matrix_fragment_71, matrix_fragment_60,
                             matrix_fragment_61);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0x400u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_24);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_25);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_24 >> 16);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0x600u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_72, matrix_fragment_73, matrix_fragment_12,
                             matrix_fragment_13);

    load_four_shared_words<CheckBounds>(shared_tile, address_word_2 + 0x200u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_25 >> 16);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_113));

    multiply_accumulate_half(matrix_fragment_68, matrix_fragment_69, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_68, matrix_fragment_69, matrix_accumulator_116,
                             matrix_accumulator_117);

    address_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                         select_half_operand<2, false, false>(matrix_fragment_37));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_74, matrix_fragment_75, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_24 = unpack_fp8_bits(matrix_fragment_26);

    matrix_fragment_25 = unpack_fp8_bits(matrix_fragment_27);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_26 = unpack_fp8_bits(matrix_fragment_26 >> 16);

    matrix_fragment_27 = unpack_fp8_bits(matrix_fragment_27 >> 16);

    multiply_accumulate_half(matrix_fragment_74, matrix_fragment_75, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_36));

    address_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(address_word_2));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_50 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(matrix_accumulator_106));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    uniform_index_word_4 = blockIdx.y;

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                       select_half_operand<2, false, false>(address_word_2));

    address_word_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                         select_half_operand<2, false, false>(matrix_fragment_68));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_fragment_70));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_69));

    matrix_fragment_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_71));

    address_word_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(matrix_accumulator_108),
                                      select_half_operand<2, false, false>(address_word_2));

    matrix_fragment_25 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_24 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(matrix_accumulator_109),
                                      select_half_operand<2, false, false>(matrix_fragment_24));

    matrix_fragment_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_26));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                     select_half_operand<2, false, false>(address_word_2));

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_24));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_48,
                             matrix_fragment_49);

    uniform_condition_0 = int32_t(uniform_parameter_word_5) >= int32_t(uniform_index_word_10);

    uniform_index_word_4 = (uniform_index_word_4 << 0x3u) + uniform_index_word_8;

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_27 = result_word;
    }

    matrix_fragment_24 = parameters[parameter_output_low_2];
    matrix_fragment_25 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_112), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_33 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_26), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_77 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_60,
                             matrix_fragment_61);

    uniform_condition_0 =
        int32_t(uniform_index_word_4) > int32_t(0u - 0x4u) && !uniform_condition_0;

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_62,
                             matrix_fragment_63);

    uniform_condition_3 = uniform_condition_0 & uniform_condition_2;

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                       select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(matrix_fragment_33));

    condition_4 = uniform_condition_3;

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_77));

    multiply_accumulate_half(matrix_accumulator_80, matrix_accumulator_81, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_80,
                             matrix_accumulator_81);

    uniform_index_word_8 =
        uniform_parameter_word_5 * uniform_index_word_11 + uniform_address_word_9;

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_122 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_82, matrix_accumulator_83, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_accumulator_82,
                             matrix_accumulator_83);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_112), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_113 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                       select_half_operand<2, false, false>(0u));

    address_word_2 = packed_half_161 * 0x80u;

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_29 = uniform_index_word_8;

    if (condition_4) {
        matrix_fragment_30 = permute_signed(matrix_fragment_30, matrix_fragment_27, 0x5410u);
    }

    uniform_condition_0 = uniform_condition_0 & uniform_condition_1;

    matrix_fragment_27 = matrix_fragment_29 * 0x100u + address_word_2;

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                             matrix_fragment_41);

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        int64_t(int32_t(matrix_fragment_27)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_24, matrix_fragment_25));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_83),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_82),
                                       select_half_operand<2, false, false>(0u));

    if (condition_4) {
        split_address_words(matrix_accumulator_118, matrix_accumulator_119,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));

        matrix_fragment_31 = permute_signed(matrix_fragment_32, matrix_fragment_31, 0x5410u);

        matrix_fragment_28 = permute_signed(matrix_fragment_28, matrix_fragment_33, 0x5410u);

        matrix_fragment_29 = permute_signed(matrix_fragment_29, matrix_fragment_26, 0x5410u);
    }
    if (uniform_condition_0) {
        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_75),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_74),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_26 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                          select_half_operand<2, false, false>(matrix_fragment_14));

        matrix_fragment_27 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                          select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_15 = matrix_fragment_26;

        matrix_fragment_13 = matrix_fragment_27;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }
    if (condition_4) {
        matrix_fragment_12 = matrix_fragment_31;

        matrix_fragment_13 = matrix_fragment_30;

        matrix_fragment_14 = matrix_fragment_29;

        matrix_fragment_15 = matrix_fragment_28;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_118, matrix_accumulator_119) + 0x400ull,
            matrix_fragment_12, matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
            memory_ranges, error_counts);
    }

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xb6a0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xb0a0ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xb2a0ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xb4a0ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xbaa0ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xb8a0ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xbca0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_16, matrix_accumulator_17) + 0xbea0ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                       select_half_operand<2, false, false>(matrix_accumulator_122));

    matrix_accumulator_17 = permute_signed(matrix_accumulator_116, matrix_accumulator_116, 0x5432u);

    packed_half_16 = permute_signed(matrix_accumulator_115, matrix_accumulator_115, 0x5432u);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                       select_half_operand<2, false, false>(packed_half_16));

    matrix_accumulator_116 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x4100410u);

    matrix_accumulator_115 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x4100410u);

    matrix_accumulator_17 = half_to_float_bits((matrix_accumulator_116 & 65535u) * 65537u);

    matrix_accumulator_116 = half_to_float_bits((matrix_accumulator_116 >> 16) * 65537u);

    matrix_accumulator_117 = half_to_float_bits((matrix_accumulator_115 & 65535u) * 65537u);

    matrix_accumulator_115 = half_to_float_bits((matrix_accumulator_115 >> 16) * 65537u);

    matrix_accumulator_17 = rsqrt_bits(matrix_accumulator_17);

    matrix_accumulator_116 = rsqrt_bits(matrix_accumulator_116);

    matrix_accumulator_117 = rsqrt_bits(matrix_accumulator_117);

    packed_half_16 = rsqrt_bits(matrix_accumulator_115);

    matrix_accumulator_118 = pack_half(matrix_accumulator_116, matrix_accumulator_17);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    packed_half_16 = pack_half(packed_half_16, matrix_accumulator_117);

    matrix_accumulator_101 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(packed_half_16));

    matrix_accumulator_100 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_101 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<2, false, false>(packed_half_16));

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_100 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                      select_half_operand<2, false, false>(0u));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = permute_signed(matrix_accumulator_101, packed_half_16, 0x5410u);

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<2, false, false>(matrix_accumulator_118));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    packed_half_16 = permute_signed(matrix_fragment_77, matrix_fragment_77, 0x5432u);

    matrix_fragment_36 = permute_signed(matrix_accumulator_100, matrix_accumulator_17, 0x5410u);

    matrix_accumulator_17 = permute_signed(packed_half_112, packed_half_112, 0x5432u);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_106),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_107),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(packed_half_16));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(matrix_accumulator_17));

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_107),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x4100410u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0x4100410u);

    matrix_fragment_39 = permute_signed(matrix_fragment_39, matrix_accumulator_106, 0x5410u);

    matrix_accumulator_106 = half_to_float_bits((matrix_fragment_77 & 65535u) * 65537u);

    matrix_fragment_77 = half_to_float_bits((matrix_fragment_77 >> 16) * 65537u);

    matrix_accumulator_100 = half_to_float_bits((packed_half_112 & 65535u) * 65537u);

    packed_half_112 = half_to_float_bits((packed_half_112 >> 16) * 65537u);

    matrix_accumulator_106 = rsqrt_bits(matrix_accumulator_106);

    matrix_fragment_77 = rsqrt_bits(matrix_fragment_77);

    matrix_accumulator_100 = rsqrt_bits(matrix_accumulator_100);

    matrix_accumulator_101 = rsqrt_bits(packed_half_112);

    matrix_fragment_38 = permute_signed(matrix_fragment_38, matrix_accumulator_115, 0x5410u);

    matrix_accumulator_100 = pack_half(matrix_accumulator_101, matrix_accumulator_100);

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_fragment_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_68 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_68),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_16, matrix_accumulator_17, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 = pack_half(matrix_fragment_77, matrix_accumulator_106);

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_accumulator_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_accumulator_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<0, false, false>(packed_half_11));

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_69),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_71),
                            select_half_operand<0, false, false>(packed_half_11));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_109),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                                       select_half_operand<2, false, false>(0u));

    packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_69 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_69),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_fragment_12 = permute_signed(matrix_fragment_12, packed_half_11, 0x5410u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_150, matrix_fragment_151, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_42,
                             matrix_fragment_43);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_70),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_71),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = permute_signed(matrix_fragment_13, matrix_fragment_14, 0x5410u);

    matrix_fragment_14 = permute_signed(matrix_fragment_68, matrix_fragment_15, 0x5410u);

    matrix_fragment_15 = permute_signed(matrix_fragment_69, matrix_fragment_38, 0x5410u);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_34,
                             matrix_fragment_35);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_150, matrix_fragment_151, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_48,
                             matrix_fragment_49);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_44, matrix_fragment_45, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_29 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_29),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_31 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_31),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_46, matrix_fragment_47, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_28 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_28),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_30 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_30),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_34),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x3c203c20u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_31 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x3c203c20u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_28 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_30 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x3c203c20u);

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3e473e47u);

    matrix_fragment_31 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x3e473e47u);

    matrix_fragment_34 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3c203c20u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_84, matrix_fragment_85, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x3e473e47u);

    matrix_fragment_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x3e473e47u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3c203c20u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3c203c20u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x3e473e47u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_33),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x3e473e47u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_20 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_21 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_30 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = (matrix_fragment_31 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3e473e47u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    matrix_accumulator_19 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_28 = (matrix_fragment_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_29 = (matrix_fragment_29 << 0x5u) + 0x7ff88000u;

    matrix_fragment_32 = (matrix_fragment_9 << 0x5u) + 0x7ff88000u;

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3c203c20u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_accumulator_18 = (matrix_fragment_34 << 0x5u) + 0x7ff88000u;

    matrix_fragment_34 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_8 = (matrix_fragment_48 << 0x5u) + 0x7ff88000u;

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_9 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_35),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_11 = (packed_half_11 << 0x5u) + 0x7ff88000u;

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                                        select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3e473e47u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3c203c20u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3c203c20u);

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3c203c20u);

    matrix_fragment_44 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_35 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3c203c20u);

    matrix_fragment_38 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x3e473e47u);

    matrix_fragment_39 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                       select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_fragment_34 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x3e473e47u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_24 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_33 = (matrix_fragment_42 << 0x5u) + 0x7ff88000u;

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3c203c20u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3e473e47u);

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x3e473e47u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_fragment_40 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x3e473e47u);

    packed_half_16 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_16),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3c203c20u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_25),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3c203c20u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_35 = (matrix_fragment_43 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_26 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_17 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3e473e47u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3e473e47u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_34 = (matrix_fragment_34 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_39 = (matrix_fragment_39 << 0x5u) + 0x7ff88000u;

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    packed_half_16 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_16), 0x3c203c20u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_25 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3c203c20u);

    packed_half_3 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<0, false, false>(packed_half_3), 0x3d343d34u);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3c203c20u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_38 = (matrix_fragment_49 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_41 = (matrix_fragment_51 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_17 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17), 0x3c203c20u);

    packed_half_16 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_16), 0x3e473e47u);

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3c203c20u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x3e473e47u);

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x3e473e47u);

    packed_half_3 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x3c203c20u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3c203c20u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_36 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17), 0x3e473e47u);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3e473e47u);

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_3), 0x3e473e47u);

    matrix_accumulator_17 = (packed_half_16 << 0x5u) + 0x7ff88000u;

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x3e473e47u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    packed_half_3 = (matrix_fragment_25 << 0x5u) + 0x7ff88000u;

    packed_half_16 = (matrix_fragment_27 << 0x5u) + 0x7ff88000u;

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_42 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_fragment_36 = (matrix_fragment_36 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_37 = (matrix_fragment_37 << 0x5u) + 0x7ff88000u;

    matrix_fragment_25 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_27 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                        select_half_operand<2, false, false>(matrix_fragment_37));

    matrix_fragment_47 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                        select_half_operand<2, false, false>(matrix_fragment_44));

    matrix_fragment_49 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                        select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_48 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_13 = condition_0 ? matrix_fragment_15 : matrix_fragment_12;

    matrix_fragment_12 = condition_0 ? matrix_fragment_12 : matrix_fragment_15;

    matrix_fragment_14 = condition_0 ? matrix_fragment_48 : matrix_fragment_45;

    matrix_fragment_45 = condition_0 ? matrix_fragment_45 : matrix_fragment_48;

    matrix_fragment_15 = !condition_1 ? matrix_fragment_13 : matrix_fragment_14;

    matrix_fragment_47 = !condition_1 ? matrix_fragment_12 : matrix_fragment_45;

    matrix_fragment_14 = !condition_1 ? matrix_fragment_14 : matrix_fragment_13;

    matrix_fragment_45 = !condition_1 ? matrix_fragment_45 : matrix_fragment_12;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_15), "r"(matrix_fragment_76), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(matrix_accumulator_134), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_134 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_accumulator_121), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_14 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_45), "r"(matrix_accumulator_120), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    matrix_fragment_12 = !condition_2 ? matrix_fragment_15 : matrix_accumulator_134;

    matrix_fragment_13 = !condition_2 ? matrix_accumulator_134 : matrix_fragment_15;

    matrix_fragment_49 = !condition_2 ? matrix_fragment_14 : matrix_fragment_45;

    matrix_fragment_48 = !condition_2 ? matrix_fragment_45 : matrix_fragment_14;

    matrix_fragment_50 = !condition_3 ? matrix_fragment_12 : matrix_fragment_49;

    matrix_fragment_51 = !condition_3 ? matrix_fragment_13 : matrix_fragment_48;

    matrix_fragment_12 = !condition_3 ? matrix_fragment_49 : matrix_fragment_12;

    matrix_fragment_13 = !condition_3 ? matrix_fragment_48 : matrix_fragment_13;

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_50 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_50),
                                        select_half_operand<1, false, false>(matrix_fragment_50));

    matrix_fragment_50 = permute_signed(matrix_fragment_50, matrix_fragment_50, 0x5410u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x4100410u);

    matrix_fragment_12 = half_to_float_bits((matrix_fragment_50 & 65535u) * 65537u);

    matrix_fragment_50 = half_to_float_bits((matrix_fragment_50 >> 16) * 65537u);

    matrix_fragment_12 = reciprocal_bits(matrix_fragment_12);

    matrix_fragment_13 = reciprocal_bits(matrix_fragment_50);

    matrix_fragment_58 =
        uint32_t(combine_words(matrix_fragment_58, 0u) << 0x5u >> 32) + matrix_fragment_57;

    matrix_fragment_15 = uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_58 = matrix_fragment_58 & 0xffffffe0u;

    matrix_fragment_45 = uint32_t(int64_t(int32_t(index_word_0)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_48 = uint32_t(int32_t(matrix_fragment_45) >> 0x1fu);

    matrix_fragment_14 = pack_half(matrix_fragment_13, matrix_fragment_12);

    matrix_fragment_13 = matrix_fragment_57 + (0u - matrix_fragment_58);

    matrix_fragment_12 = uint32_t(int32_t(matrix_fragment_15) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_59), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_59 = result_word;
    }

    matrix_fragment_48 =
        uint32_t(combine_words(matrix_fragment_48, 0u) << 0x5u >> 32) + matrix_fragment_45;

    matrix_fragment_12 =
        uint32_t(combine_words(matrix_fragment_12, 0u) << 0x5u >> 32) + matrix_fragment_15;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_48 = matrix_fragment_48 & 0xffffffe0u;

    matrix_fragment_12 = matrix_fragment_12 & 0xffffffe0u;

    matrix_fragment_49 = matrix_fragment_45 + (0u - matrix_fragment_48);

    matrix_fragment_47 = matrix_fragment_15 + (0u - matrix_fragment_12);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_49), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(matrix_fragment_47), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    matrix_fragment_59 = permute_signed(matrix_fragment_59, matrix_fragment_59, 0x5410u);

    matrix_fragment_45 = permute_signed(matrix_fragment_13, matrix_fragment_13, 0x5410u);

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_accumulator_20 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = permute_signed(matrix_fragment_50, matrix_fragment_50, 0x5410u);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_accumulator_19));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_28));

    matrix_fragment_47 = permute_signed(matrix_fragment_47, matrix_fragment_47, 0x5410u);

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    packed_half_11 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                                         select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_52, matrix_fragment_53, 0u, 0u);

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                          select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    packed_half_16 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                                         select_half_operand<2, false, false>(matrix_fragment_50));

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_22, matrix_accumulator_23, 0u, 0u);

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(0u));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_3),
                                        select_half_operand<2, false, false>(matrix_fragment_50));

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = parameters[parameter_trained_weights_low_4];
    matrix_fragment_9 = parameters[parameter_trained_weights_high_5];

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_98, matrix_fragment_99, matrix_fragment_30,
                             matrix_fragment_31);

    matrix_fragment_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                       select_half_operand<2, false, false>(packed_half_16));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_46));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_102, matrix_fragment_103, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_104,
                             matrix_fragment_105);

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_98, matrix_fragment_99, matrix_accumulator_22,
                             matrix_accumulator_23);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_14 = index_word_0 & 0x7ffffffcu;

    matrix_fragment_12 = index_word_0 & 0xfffffffcu;

    packed_half_11 = 0u - matrix_fragment_14 + address_word_157;

    matrix_accumulator_17 = matrix_fragment_4 + address_word_157 + (0u - matrix_fragment_12);

    matrix_fragment_25 = matrix_fragment_5 + address_word_157 + (0u - matrix_fragment_12);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    packed_half_11 = packed_half_11 * 0x2u + packed_half_10;

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_19 = matrix_fragment_6 + address_word_157 + (0u - matrix_fragment_12);

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(0u));

    packed_half_11 = uint32_t(int32_t(packed_half_11) >> 0x1u);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_28, matrix_fragment_29,
                        uint64_t(uint32_t(matrix_accumulator_17)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    split_address_words(matrix_fragment_4, matrix_fragment_5,
                        int64_t(int32_t(packed_half_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                                       select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_28, matrix_fragment_29) + 0xf0b0ull, memory_ranges,
        error_counts);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    split_address_words(matrix_fragment_30, matrix_fragment_31,
                        uint64_t(uint32_t(matrix_accumulator_19)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_4, matrix_fragment_5) + 0xf0b0ull, memory_ranges,
        error_counts);

    packed_half_3 = address_word_157 * 0x10u + matrix_fragment_7;

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        uint64_t(uint32_t(matrix_fragment_25)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_fragment_8, matrix_fragment_9));

    matrix_fragment_42 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_30, matrix_fragment_31) + 0xf0b0ull, memory_ranges,
        error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_3, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, packed_half_3 + 0x400u, matrix_fragment_52,
                                         matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                         shared_bytes, memory_ranges, error_counts);

    matrix_fragment_43 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0xf0b0ull, memory_ranges,
        error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_3 + 0x800u, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_3 + 0xc00u, packed_half_16, matrix_accumulator_17,
        matrix_accumulator_18, matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xe0b0ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xe2b0ull, matrix_accumulator_20,
        matrix_accumulator_21, matrix_accumulator_22, matrix_accumulator_23, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xe8b0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0xeab0ull, matrix_fragment_4,
        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);

    uniform_index_word_12 = 0u;

    uniform_index_word_8 = 0x400u;

    uniform_index_word_8 = (uniform_index_word_12 << 0x18u) + uniform_index_word_8;

    matrix_fragment_8 = (address_word_157 << 0x4u) + uniform_index_word_8;

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_36);

    packed_half_11 = unpack_fp8_bits(matrix_fragment_37);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_36 >> 16);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_37 >> 16);

    uniform_index_word_11 =
        uniform_parameter_word_5 * uniform_index_word_11 + uniform_index_word_11;

    uniform_index_word_11 = uniform_address_word_9 + uniform_index_word_11;

    condition_0 = (address_word_157 & 0x4u) != 0;

    uniform_parameter_word_5 = uniform_parameter_word_5 + 0x1u;

    uniform_condition_0 = int32_t(uniform_parameter_word_5) >= int32_t(uniform_index_word_10);

    uniform_condition_0 =
        int32_t(uniform_index_word_4) > int32_t(0u - 0x8u) && !uniform_condition_0;

    uniform_condition_2 = uniform_condition_0 & uniform_condition_2;

    uniform_condition_0 = uniform_condition_0 & uniform_condition_1;

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_38);

    packed_half_11 = unpack_fp8_bits(matrix_fragment_39);

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_38 >> 16);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_39 >> 16);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_28,
                             matrix_fragment_29);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8 + 0x400u, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_9 = unpack_fp8_bits(packed_half_16);

    packed_half_11 = unpack_fp8_bits(matrix_accumulator_17);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_38,
                             matrix_fragment_39);

    packed_half_16 = unpack_fp8_bits(packed_half_16 >> 16);

    matrix_accumulator_17 = unpack_fp8_bits(matrix_accumulator_17 >> 16);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_36,
                             matrix_fragment_37);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8 + 0x200u, matrix_fragment_36,
                                        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_32,
                             matrix_fragment_33);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8 + 0x600u, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_16 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_16),
                                         select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_17),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_9 = unpack_fp8_bits(matrix_accumulator_18);

    packed_half_11 = unpack_fp8_bits(matrix_accumulator_19);

    matrix_accumulator_18 = unpack_fp8_bits(matrix_accumulator_18 >> 16);

    matrix_accumulator_19 = unpack_fp8_bits(matrix_accumulator_19 >> 16);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_26, matrix_fragment_27, packed_half_16,
                             matrix_accumulator_17);

    matrix_accumulator_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_18),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_16 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                         select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_11),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_19),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_20, matrix_accumulator_21, packed_half_16,
                             matrix_accumulator_17);

    packed_half_11 = uniform_index_word_11;

    matrix_fragment_9 = uint32_t(uint32_t(address_word_157) >> 0x1u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_22, matrix_accumulator_23, matrix_accumulator_18,
                             matrix_accumulator_19);

    packed_half_11 = packed_half_11 * 0x100u + address_word_2;

    address_word_2 = !condition_0 ? matrix_fragment_72 : matrix_fragment_88;

    matrix_accumulator_22 = address_word_157 & 0x13u;

    multiply_accumulate_half(packed_half_16, matrix_accumulator_17, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_accumulator_22 = matrix_accumulator_22 | 0x4u & matrix_fragment_9;

    matrix_fragment_9 = address_word_157 * 0x2u;

    matrix_accumulator_23 = !condition_0 ? matrix_fragment_73 : matrix_fragment_89;

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_24,
                             matrix_fragment_25);

    matrix_fragment_9 = matrix_accumulator_22 | 0x8u & matrix_fragment_9;

    matrix_fragment_30 = !condition_3 ? matrix_accumulator_23 : address_word_2;

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_24 = !condition_0 ? matrix_fragment_74 : matrix_fragment_90;

    matrix_fragment_25 = !condition_0 ? matrix_fragment_75 : matrix_fragment_91;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_26,
                             matrix_fragment_27);

    matrix_fragment_26 = !condition_3 ? address_word_2 : matrix_accumulator_23;

    multiply_accumulate_half(matrix_accumulator_18, matrix_accumulator_19, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_48,
                             matrix_fragment_49);

    matrix_fragment_42 = !condition_3 ? matrix_fragment_24 : matrix_fragment_25;

    matrix_fragment_44 = !condition_3 ? matrix_fragment_25 : matrix_fragment_24;

    matrix_fragment_24 = !condition_0 ? matrix_accumulator_94 : matrix_accumulator_82;

    matrix_fragment_25 = !condition_0 ? matrix_accumulator_95 : matrix_accumulator_83;

    address_word_2 = !condition_0 ? matrix_accumulator_92 : matrix_accumulator_80;

    multiply_accumulate_half(matrix_accumulator_22, matrix_accumulator_23, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_4, matrix_fragment_5, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_64 = !condition_3 ? matrix_fragment_24 : matrix_fragment_25;

    matrix_fragment_66 = !condition_3 ? matrix_fragment_25 : matrix_fragment_24;

    matrix_fragment_24 = parameters[parameter_output_low_2];
    matrix_fragment_25 = parameters[parameter_output_high_3];

    matrix_fragment_5 = !condition_0 ? matrix_accumulator_93 : matrix_accumulator_81;

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_36,
                             matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_50,
                             matrix_fragment_51);

    matrix_fragment_52 = !condition_3 ? address_word_2 : matrix_fragment_5;

    matrix_fragment_54 = !condition_3 ? matrix_fragment_5 : address_word_2;

    matrix_fragment_4 = !condition_0 ? matrix_fragment_12 : packed_half_16;

    address_word_2 = !condition_0 ? packed_half_16 : matrix_fragment_12;

    matrix_fragment_5 = !condition_0 ? matrix_accumulator_17 : matrix_fragment_13;

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_38 = !condition_3 ? matrix_fragment_5 : address_word_2;

    matrix_fragment_7 = !condition_0 ? matrix_fragment_13 : matrix_accumulator_17;

    matrix_fragment_34 = !condition_3 ? address_word_2 : matrix_fragment_5;

    address_word_2 = !condition_0 ? matrix_accumulator_20 : matrix_fragment_14;

    matrix_fragment_5 = !condition_0 ? matrix_accumulator_21 : matrix_fragment_15;

    matrix_fragment_31 = !condition_3 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_40 = !condition_3 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_4 = !condition_0 ? matrix_fragment_14 : matrix_accumulator_20;

    matrix_fragment_7 = !condition_0 ? matrix_fragment_15 : matrix_accumulator_21;

    matrix_fragment_46 = !condition_3 ? address_word_2 : matrix_fragment_5;

    matrix_fragment_48 = !condition_3 ? matrix_fragment_5 : address_word_2;

    address_word_2 = !condition_0 ? matrix_accumulator_18 : matrix_accumulator_22;

    matrix_fragment_5 = !condition_0 ? matrix_accumulator_19 : matrix_accumulator_23;

    matrix_fragment_33 = !condition_3 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_50 = !condition_3 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_4 = !condition_0 ? matrix_accumulator_22 : matrix_accumulator_18;

    matrix_fragment_7 = !condition_0 ? matrix_accumulator_23 : matrix_accumulator_19;

    matrix_fragment_72 = !condition_0 ? matrix_fragment_88 : matrix_fragment_72;

    matrix_fragment_73 = !condition_0 ? matrix_fragment_89 : matrix_fragment_73;

    matrix_fragment_74 = !condition_0 ? matrix_fragment_90 : matrix_fragment_74;

    matrix_fragment_75 = !condition_0 ? matrix_fragment_91 : matrix_fragment_75;

    matrix_accumulator_80 = !condition_0 ? matrix_accumulator_80 : matrix_accumulator_92;

    matrix_accumulator_81 = !condition_0 ? matrix_accumulator_81 : matrix_accumulator_93;

    matrix_accumulator_82 = !condition_0 ? matrix_accumulator_82 : matrix_accumulator_94;

    matrix_accumulator_83 = !condition_0 ? matrix_accumulator_83 : matrix_accumulator_95;

    matrix_fragment_35 = !condition_3 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_62 = !condition_3 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_58 = !condition_3 ? address_word_2 : matrix_fragment_5;

    matrix_fragment_60 = !condition_3 ? matrix_fragment_5 : address_word_2;

    matrix_fragment_4 = !condition_0 ? matrix_fragment_36 : matrix_fragment_28;

    matrix_fragment_7 = !condition_0 ? matrix_fragment_37 : matrix_fragment_29;

    address_word_2 = !condition_0 ? matrix_fragment_28 : matrix_fragment_36;

    matrix_fragment_5 = !condition_0 ? matrix_fragment_29 : matrix_fragment_37;

    matrix_fragment_27 = !condition_3 ? matrix_fragment_72 : matrix_fragment_73;

    matrix_fragment_39 = !condition_3 ? matrix_fragment_74 : matrix_fragment_75;

    matrix_fragment_41 = !condition_3 ? matrix_accumulator_80 : matrix_accumulator_81;

    matrix_fragment_43 = !condition_3 ? matrix_accumulator_82 : matrix_accumulator_83;

    matrix_fragment_72 = !condition_3 ? matrix_fragment_73 : matrix_fragment_72;

    matrix_fragment_74 = !condition_3 ? matrix_fragment_75 : matrix_fragment_74;

    matrix_accumulator_80 = !condition_3 ? matrix_accumulator_81 : matrix_accumulator_80;

    matrix_accumulator_82 = !condition_3 ? matrix_accumulator_83 : matrix_accumulator_82;

    matrix_fragment_68 = !condition_3 ? matrix_fragment_4 : matrix_fragment_7;

    matrix_fragment_70 = !condition_3 ? matrix_fragment_7 : matrix_fragment_4;

    matrix_fragment_45 = !condition_3 ? address_word_2 : matrix_fragment_5;

    matrix_fragment_76 = !condition_3 ? matrix_fragment_5 : address_word_2;

    matrix_fragment_32 = 0x4u ^ matrix_fragment_9;

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        int64_t(int32_t(packed_half_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_24, matrix_fragment_25));
    if (uniform_condition_0) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                          select_half_operand<2, false, false>(0u));

        address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_5 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_4 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                       select_half_operand<2, false, false>(address_word_2));

        packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_16),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

        split_address_words(packed_half_16, matrix_accumulator_17,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_18),
                          select_half_operand<2, false, false>(matrix_fragment_7));

        matrix_fragment_7 = matrix_fragment_4;

        matrix_fragment_4 = packed_half_11;

        matrix_fragment_5 = address_word_2;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_16, matrix_accumulator_17), matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    if (uniform_condition_2) {
        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(matrix_fragment_24, matrix_fragment_25,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(0u));

        address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

        packed_half_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

        matrix_fragment_7 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                          select_half_operand<2, false, false>(matrix_fragment_4));

        matrix_fragment_6 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_22),
                          select_half_operand<2, false, false>(matrix_fragment_5));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(address_word_2));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(packed_half_11));

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_24, matrix_fragment_25) + 0x400ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    matrix_fragment_5 = 0x10u ^ matrix_fragment_9;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_39), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    matrix_fragment_7 = 0x14u ^ matrix_fragment_9;

    uniform_block_index_13 = blockIdx.x;

    uniform_parameter_word_5 = parameters[parameter_optional_control_10];

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_42), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_accumulator_22 = 0u;

    uniform_index_word_4 =
        uint32_t(combine_words(uniform_index_word_4, 0u) << 0x1u >> 32) + uniform_index_word_4;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_74), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_17 = result_word;
    }

    uniform_index_word_7 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x1u >> 32) + uniform_index_word_7;

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    uniform_index_word_4 = uint32_t(int32_t(uniform_index_word_4) >> 0x1u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_44), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_44 = result_word;
    }

    uniform_index_word_7 = uint32_t(int32_t(uniform_index_word_7) >> 0x1u);

    uniform_index_word_6 = uint32_t(int32_t(uniform_index_word_6) >> 0x1u);

    uniform_address_word_9 = uniform_index_word_4 + 0x2u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_27), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_27 = result_word;
    }

    uniform_index_word_8 = uniform_index_word_7 << (0x2u & 31u);

    uniform_condition_0 = true;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_26), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_26 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_30), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_30 = result_word;
    }

    uniform_parameter_word_5 = (uniform_block_index_13 << 0x3u) + uniform_parameter_word_5;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        packed_half_11 = result_word;
    }

    matrix_fragment_39 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_39));

    uniform_parameter_word_5 = uint32_t(combine_words(uniform_parameter_word_5, 0u) << 0x1u >> 32) +
                               uniform_parameter_word_5;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_33), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_33 = result_word;
    }

    uniform_parameter_word_5 = uint32_t(int32_t(uniform_parameter_word_5) >> 0x1u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_43), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    matrix_accumulator_17 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_45), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_45 = result_word;
    }

    matrix_accumulator_17 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                       select_half_operand<2, false, false>(matrix_accumulator_17));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_46), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_26 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_27));

    matrix_accumulator_17 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_17), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_64 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_68), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_68 = result_word;
    }

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                        select_half_operand<2, false, false>(packed_half_11));

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_17),
                                   select_half_operand<2, false, false>(0u));

    packed_half_11 = packed_half_161 * 0x2u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_50), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_19 = result_word;
    }

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_48), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_82), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_23 = result_word;
    }

    packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                   select_half_operand<2, false, false>(address_word_2));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_66), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_70), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_70 = result_word;
    }

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_25 = result_word;
    }

    matrix_fragment_45 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_68),
                                        select_half_operand<2, false, false>(matrix_fragment_45));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_34), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_34 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_52), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_52 = result_word;
    }

    matrix_accumulator_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                       select_half_operand<2, false, false>(matrix_accumulator_19));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_58), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_58 = result_word;
    }

    matrix_accumulator_19 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                       select_half_operand<2, false, false>(matrix_accumulator_19));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_31), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    matrix_accumulator_23 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                       select_half_operand<2, false, false>(matrix_accumulator_23));

    matrix_accumulator_19 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_19), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_41), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    matrix_accumulator_23 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                       select_half_operand<2, false, false>(matrix_accumulator_23));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_35), "r"(matrix_fragment_32), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_70),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_19),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_23 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_23), 0x34003400u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_40), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    matrix_fragment_25 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                        select_half_operand<2, false, false>(matrix_fragment_25));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_38), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_38 = result_word;
    }

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x34003400u);

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_23),
                      select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_80), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_21 = result_word;
    }

    address_word_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                   select_half_operand<2, false, false>(0u));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_54), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_54 = result_word;
    }

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_62), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_fragment_41 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_60), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_60 = result_word;
    }

    matrix_fragment_35 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x34003400u);

    matrix_accumulator_21 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(matrix_fragment_4));

    __syncthreads();

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_21 = multiply_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x34003400u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_18 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(matrix_accumulator_18));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x34003400u);

    matrix_accumulator_20 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_21 = parameters[parameter_trained_weights_high_5];

    matrix_accumulator_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                          select_half_operand<2, false, false>(address_word_2));

    address_word_2 =
        uint32_t(combine_words(matrix_fragment_56, 0u) << 0x2u >> 32) + matrix_fragment_56;

    matrix_fragment_9 = address_word_2 & 0xfffffffcu;

    address_word_2 = address_word_157;

    matrix_fragment_56 = matrix_fragment_56 + uniform_parameter_word_5 + (0u - matrix_fragment_9);

    matrix_accumulator_23 =
        uint32_t(int64_t(int32_t(address_word_2)) * (1ll << 0x1cu) >> 32) + uniform_index_word_4;

    __syncthreads();

    condition_1 = int32_t(matrix_fragment_56) >= int32_t(uniform_index_word_7);

    condition_0 = int32_t(matrix_accumulator_23) >= int32_t(uniform_index_word_6);

    condition_1 = int32_t(matrix_fragment_56) < int32_t(0u) || condition_1;

    condition_0 = int32_t(matrix_accumulator_23) < int32_t(0u) || condition_0;

    condition_0 = condition_0 | condition_1;

    store_four_shared_words<CheckBounds>(
        shared_tile, packed_half_3, packed_half_16, matrix_accumulator_17, matrix_accumulator_18,
        matrix_accumulator_19, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    matrix_accumulator_18 = parameters[parameter_output_downsample_optional_low_18];
    matrix_accumulator_19 = parameters[parameter_output_downsample_optional_high_19];

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8, matrix_fragment_4,
                                        matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, matrix_fragment_8 + 0x200u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);
    do {
        address_word_2 = packed_half_10 + matrix_accumulator_22;

        matrix_accumulator_17 = address_word_2 * 0x8u;

        split_address_words(address_word_2, packed_half_3,
                            uint64_t(uint32_t(matrix_accumulator_17)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));

        matrix_fragment_9 = matrix_accumulator_17 + 0x80u;

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            uint64_t(uint32_t(matrix_fragment_9)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));

        split_address_words(address_word_2, packed_half_3,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(address_word_2, packed_half_3));

        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_2, packed_half_3) + 0xf130ull, matrix_fragment_24,
            matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9) + 0xf130ull, matrix_fragment_28,
            matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
            error_counts);

        matrix_accumulator_17 = matrix_accumulator_17 + 0x480u;

        load_four_words_cached<CheckBounds>(
            combine_words(address_word_2, packed_half_3) + 0x10130ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        split_address_words(packed_half_16, matrix_accumulator_17,
                            int64_t(int32_t(matrix_accumulator_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_20, matrix_accumulator_21));

        split_address_words(packed_half_16, matrix_accumulator_17,
                            int64_t(int32_t(address_word_157)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_16, matrix_accumulator_17));

        load_four_words_cached<CheckBounds>(
            combine_words(packed_half_16, matrix_accumulator_17) + 0xf130ull, matrix_fragment_36,
            matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges,
            error_counts);

        uniform_condition_1 = uniform_condition_0;

        matrix_accumulator_22 =
            uint32_t(combine_words(matrix_accumulator_22, 0u) << 0x1cu >> 32) + packed_half_11;

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_24, matrix_fragment_25, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_26, matrix_fragment_27, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_28, matrix_fragment_29, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                                 matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                                 matrix_fragment_30, matrix_fragment_31, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_24,
                                 matrix_fragment_25);

        multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_26,
                                 matrix_fragment_27);

        multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                                 matrix_fragment_29);

        multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_38, matrix_fragment_39, matrix_fragment_30,
                                 matrix_fragment_31);
        if (!condition_0) {
            matrix_fragment_56 = matrix_fragment_56 * 0x4u;

            packed_half_3 = index_word_0 & 0xfffffffcu;

            matrix_accumulator_23 =
                matrix_accumulator_22 * uniform_index_word_6 + matrix_accumulator_23;

            matrix_fragment_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                              select_half_operand<2, false, false>(0u));

            matrix_fragment_56 = matrix_accumulator_23 * uniform_index_word_8 + matrix_fragment_56;

            matrix_fragment_9 =
                pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                              select_half_operand<2, false, false>(matrix_fragment_9));

            packed_half_3 = matrix_fragment_56 + address_word_157 + (0u - packed_half_3);

            split_address_words(address_word_2, packed_half_3,
                                int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_18, matrix_accumulator_19));

            pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        address_word_157 = threadIdx.x & 31u;

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_17 = matrix_accumulator_22 + 0x1u;

        packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(0u));

        index_word_0 = address_word_157;

        matrix_accumulator_23 = address_word_157;

        matrix_fragment_56 = uint32_t(int32_t(index_word_0) >> 0x2u);

        matrix_fragment_8 =
            uint32_t(int64_t(int32_t(matrix_accumulator_23)) * (1ll << 0x1cu) >> 32) +
            uniform_address_word_9;

        address_word_2 =
            uint32_t(combine_words(matrix_fragment_56, 0u) << 0x2u >> 32) + matrix_fragment_56;

        matrix_accumulator_23 =
            uint32_t(int64_t(int32_t(matrix_accumulator_23)) * (1ll << 0x1cu) >> 32) +
            uniform_index_word_4;

        packed_half_3 = address_word_2 & 0xfffffffcu;

        condition_0 = int32_t(matrix_fragment_8) >= int32_t(uniform_index_word_6);

        condition_3 = int32_t(matrix_accumulator_23) >= int32_t(uniform_index_word_6);

        matrix_fragment_56 = matrix_fragment_56 + uniform_parameter_word_5 + (0u - packed_half_3);

        condition_0 = int32_t(matrix_fragment_8) < int32_t(0u) || condition_0;

        condition_3 = int32_t(matrix_accumulator_23) < int32_t(0u) || condition_3;

        condition_1 = int32_t(matrix_fragment_56) >= int32_t(uniform_index_word_7);

        condition_2 = int32_t(matrix_fragment_56) < int32_t(0u) || condition_1;

        condition_1 = condition_0 | condition_2;

        condition_0 = condition_3 | condition_2;
        if (!condition_1) {
            packed_half_3 = matrix_fragment_56 * 0x4u;

            matrix_fragment_9 = permute_signed(matrix_fragment_9, packed_half_16, 0x5410u);

            address_word_2 = matrix_accumulator_22 * uniform_index_word_6 + matrix_fragment_8;

            address_word_2 = address_word_2 * uniform_index_word_8 + packed_half_3;

            packed_half_3 = index_word_0 & 0xfffffffcu;

            packed_half_3 = address_word_2 + address_word_157 + (0u - packed_half_3);

            split_address_words(address_word_2, packed_half_3,
                                int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_18, matrix_accumulator_19));

            pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                           select_half_operand<2, false, false>(0u));
        if (!condition_0) {
            packed_half_3 = matrix_fragment_56 * 0x4u;

            matrix_fragment_9 = permute_signed(matrix_fragment_9, matrix_fragment_30, 0x5410u);

            address_word_2 = matrix_accumulator_17 * uniform_index_word_6 + matrix_accumulator_23;

            address_word_2 = address_word_2 * uniform_index_word_8 + packed_half_3;

            packed_half_3 = index_word_0 & 0xfffffffcu;

            packed_half_3 = address_word_2 + address_word_157 + (0u - packed_half_3);

            split_address_words(address_word_2, packed_half_3,
                                int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_18, matrix_accumulator_19));

            pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(0u));

        packed_half_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));
        if (!condition_1) {
            packed_half_3 = matrix_fragment_56 * 0x4u;

            matrix_fragment_9 = permute_signed(matrix_fragment_9, packed_half_16, 0x5410u);

            matrix_fragment_8 = matrix_accumulator_17 * uniform_index_word_6 + matrix_fragment_8;

            matrix_fragment_8 = matrix_fragment_8 * uniform_index_word_8 + packed_half_3;

            packed_half_3 = index_word_0 & 0xfffffffcu;

            packed_half_3 = matrix_fragment_8 + address_word_157 + (0u - packed_half_3);

            split_address_words(address_word_2, packed_half_3,
                                int64_t(int32_t(packed_half_3)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_18, matrix_accumulator_19));

            pipeline_write<CheckBounds>(combine_words(address_word_2, packed_half_3),
                                        matrix_fragment_9, memory_ranges, error_counts);
        }
        matrix_accumulator_22 = 0x40u;

        uniform_condition_0 = false;
    } while (uniform_condition_1);
    uniform_index_word_10 = parameters[parameter_output_downsample_optional_low_18];
    uniform_index_word_11 = parameters[parameter_output_downsample_optional_high_19];

    matrix_fragment_4 = parameters[parameter_tensor_height_8];
    matrix_fragment_5 = parameters[parameter_tensor_width_9];

    condition_0 = 0ull != combine_words(uniform_index_word_10, uniform_index_word_11);

    if (!condition_0) {
        return;
    }

    uniform_index_word_6 = parameters[parameter_tensor_height_20];
    uniform_index_word_7 = parameters[parameter_tensor_width_21];

    address_word_2 = matrix_fragment_5 + 0x1u;

    index_word_0 = matrix_fragment_4 + 0x1u;

    address_word_2 = uint32_t(combine_words(address_word_2, 0u) << 0x1u >> 32) + address_word_2;

    index_word_0 = uint32_t(combine_words(index_word_0, 0u) << 0x1u >> 32) + index_word_0;

    matrix_fragment_7 = uint32_t(int32_t(address_word_2) >> 0x1u);

    matrix_fragment_12 = uint32_t(int32_t(index_word_0) >> 0x1u);

    matrix_fragment_6 = uniform_index_word_7;

    matrix_accumulator_17 = uniform_index_word_6;

    condition_0 = int32_t(matrix_fragment_7) >= int32_t(matrix_fragment_6);

    condition_1 = int32_t(matrix_fragment_12) >= int32_t(matrix_accumulator_17);

    if (condition_0) {
        return;
    }

    uniform_index_word_11 = blockDim.x;

    matrix_fragment_8 = threadIdx.z;

    address_word_2 = matrix_fragment_6 * 0x4u;

    uniform_index_word_12 = blockDim.y;

    packed_half_16 = blockDim.z;

    packed_half_11 = threadIdx.x;

    index_word_0 = address_word_2 * matrix_accumulator_17;

    uniform_index_word_6 = gridDim.y;

    uniform_index_word_8 = parameters[parameter_optional_control_10];
    uniform_address_word_9 = parameters[parameter_scalar_control_11];

    uniform_index_word_7 = gridDim.x;

    uniform_index_word_10 = uniform_index_word_4 + 0x4u;

    condition_0 = int32_t(matrix_fragment_12) < int32_t(uniform_index_word_10);

    uniform_index_word_6 = (uniform_index_word_6 << 0x3u) + uniform_address_word_9;

    condition_0 = int32_t(matrix_accumulator_17) > int32_t(uniform_index_word_4) && condition_0;

    matrix_fragment_4 = matrix_fragment_8 * uniform_index_word_12 + packed_half_161;

    uniform_index_word_7 = (uniform_index_word_7 << 0x3u) + uniform_index_word_8;

    uniform_index_word_8 = uniform_parameter_word_5 + 0x4u;

    uniform_index_word_6 = uniform_index_word_6 + (0u - 0x8u);

    uniform_index_word_7 = uniform_index_word_7 + (0u - 0x8u);

    matrix_fragment_4 = matrix_fragment_4 * uniform_index_word_11 + packed_half_11;

    uniform_index_word_6 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    condition_1 = int32_t(matrix_fragment_7) < int32_t(uniform_index_word_8);

    uniform_index_word_8 = uniform_index_word_11 * uniform_index_word_12;

    uniform_index_word_7 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x1u >> 32) + uniform_index_word_7;

    uniform_index_word_6 =
        uint32_t(int64_t(int32_t(uniform_index_word_6)) * (1ll << 0x1fu) >> 32) + 0x4u;

    condition_1 = int32_t(matrix_fragment_6) > int32_t(uniform_parameter_word_5) && condition_1;

    uniform_index_word_7 =
        uint32_t(int64_t(int32_t(uniform_index_word_7)) * (1ll << 0x1fu) >> 32) + 0x4u;

    packed_half_3 = packed_half_16 * uniform_index_word_8;

    matrix_fragment_5 = min(int32_t(matrix_accumulator_17), int32_t(uniform_index_word_6));

    condition_0 = condition_0 | condition_1;
    if (condition_0) {
        __syncthreads();

        condition_0 = uint32_t(matrix_fragment_4) > uint32_t(0xffu);

        uniform_index_word_8 = parameters[parameter_output_downsample_optional_low_18];
        uniform_address_word_9 = parameters[parameter_output_downsample_optional_high_19];

        uniform_index_word_14 = parameters[parameter_output_downsample_optional_low_18];
        uniform_address_word_15 = parameters[parameter_output_downsample_optional_high_19];
        if (!condition_0) {
            const uint32_t next_boundary_index =
                ((matrix_fragment_8 + packed_half_16) * uniform_index_word_12 + packed_half_161) *
                    uniform_index_word_11 + packed_half_11;
            matrix_fragment_14 = matrix_fragment_4;
            matrix_fragment_13 = boundary_tail_steps<false>(next_boundary_index, 256u, packed_half_3);

            matrix_fragment_8 = matrix_fragment_13 + 0x1u;

            matrix_fragment_8 = matrix_fragment_8 & 0x3u;
            condition_0 = matrix_fragment_8 != 0;
            if (condition_0) {
                matrix_accumulator_18 = parameters[parameter_tensor_height_20];
                matrix_accumulator_19 = parameters[parameter_tensor_width_21];

                matrix_fragment_15 = 0u - matrix_fragment_8;

                matrix_fragment_14 = matrix_fragment_4;
                do {
                    matrix_fragment_6 = uint32_t(uint32_t(matrix_fragment_14) >> 0x4u);

                    packed_half_11 =
                        uint32_t(combine_words(matrix_fragment_14, 0u) << 0x1au >> 32) +
                        uniform_index_word_4;

                    matrix_fragment_6 = matrix_fragment_6 & 0x3u;

                    matrix_accumulator_17 = matrix_accumulator_18;

                    matrix_fragment_15 = matrix_fragment_15 + 0x1u;

                    packed_half_10 = matrix_fragment_6 + uniform_parameter_word_5;

                    matrix_fragment_6 = matrix_accumulator_19;

                    condition_0 = int32_t(packed_half_11) >= int32_t(matrix_accumulator_17);

                    condition_2 = int32_t(packed_half_10) >= int32_t(matrix_fragment_7);

                    condition_1 = int32_t(packed_half_10) >= int32_t(matrix_fragment_6);

                    condition_0 = int32_t(packed_half_11) < int32_t(0u) || condition_0;

                    condition_2 =
                        int32_t(packed_half_11) < int32_t(matrix_fragment_12) && !condition_2;

                    condition_1 = int32_t(packed_half_10) < int32_t(0u) || condition_1;

                    condition_0 = condition_2 | condition_0 | condition_1;

                    condition_1 = uint32_t(matrix_fragment_15) != 0u;
                    if (!condition_0) {
                        matrix_fragment_9 = matrix_fragment_14 & 0xfu;

                        packed_half_10 = packed_half_10 * 0x4u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(index_word_0)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_2)) * int64_t(int32_t(packed_half_11)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        packed_half_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                                combine_words(packed_half_10, packed_half_11));

                        const uint64_t byte_address_0 = scaled_word_address(
                            combine_words(uniform_index_word_8, uniform_address_word_9),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        packed_half_10 = uint32_t(byte_address_0);

                        packed_half_11 = uint32_t(byte_address_0 >> 32);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11),
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x4ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x8ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0xcull,
                                                    0u, memory_ranges, error_counts);
                    }
                    matrix_fragment_14 = packed_half_3 + matrix_fragment_14;
                } while (condition_1);
            }
            condition_0 = uint32_t(matrix_fragment_13) >= uint32_t(0x3u);
            if (condition_0) {
                matrix_fragment_28 = parameters[parameter_tensor_height_20];
                matrix_fragment_29 = parameters[parameter_tensor_width_21];

                matrix_fragment_6 = packed_half_16 * uniform_index_word_12;

                matrix_accumulator_20 = packed_half_3 + matrix_fragment_14;

                matrix_fragment_13 = matrix_fragment_6 * uniform_index_word_11;

                packed_half_16 = matrix_fragment_13 * 0x2u + matrix_fragment_14;

                matrix_accumulator_18 = matrix_fragment_13 * 0x3u + matrix_fragment_14;
                do {
                    matrix_fragment_8 = uint32_t(uint32_t(matrix_fragment_14) >> 0x4u);

                    matrix_fragment_15 =
                        uint32_t(combine_words(matrix_fragment_14, 0u) << 0x1au >> 32) +
                        uniform_index_word_4;

                    matrix_fragment_8 = matrix_fragment_8 & 0x3u;

                    matrix_accumulator_17 = matrix_fragment_28;

                    matrix_fragment_6 = matrix_fragment_29;

                    matrix_fragment_9 = uint32_t(uint32_t(matrix_accumulator_20) >> 0x4u);

                    matrix_accumulator_19 = matrix_fragment_8 + uniform_parameter_word_5;

                    packed_half_10 = uint32_t(uint32_t(packed_half_16) >> 0x4u);

                    condition_2 = int32_t(matrix_fragment_15) >= int32_t(matrix_accumulator_17);

                    matrix_fragment_9 = matrix_fragment_9 & 0x3u;

                    condition_3 = int32_t(matrix_accumulator_19) >= int32_t(matrix_fragment_7);

                    condition_4 = int32_t(matrix_accumulator_19) >= int32_t(matrix_fragment_6);

                    packed_half_10 = packed_half_10 & 0x3u;

                    condition_2 = int32_t(matrix_fragment_15) < int32_t(0u) || condition_2;

                    condition_3 =
                        int32_t(matrix_fragment_15) < int32_t(matrix_fragment_12) && !condition_3;

                    condition_4 = int32_t(matrix_accumulator_19) < int32_t(0u) || condition_4;

                    matrix_accumulator_21 =
                        uint32_t(combine_words(matrix_accumulator_20, 0u) << 0x1au >> 32) +
                        uniform_index_word_4;

                    matrix_accumulator_23 =
                        uint32_t(combine_words(packed_half_16, 0u) << 0x1au >> 32) +
                        uniform_index_word_4;

                    matrix_accumulator_22 = matrix_fragment_9 + uniform_parameter_word_5;

                    matrix_fragment_24 = packed_half_10 + uniform_parameter_word_5;

                    packed_half_11 = uint32_t(uint32_t(matrix_accumulator_18) >> 0x4u);

                    condition_2 = condition_3 | condition_2 | condition_4;

                    condition_0 = int32_t(matrix_accumulator_21) >= int32_t(matrix_accumulator_17);

                    condition_1 = int32_t(matrix_accumulator_23) >= int32_t(matrix_accumulator_17);

                    condition_5 = int32_t(matrix_accumulator_22) >= int32_t(matrix_fragment_6);

                    condition_6 = int32_t(matrix_accumulator_22) >= int32_t(matrix_fragment_7);

                    condition_3 = int32_t(matrix_fragment_24) >= int32_t(matrix_fragment_6);

                    condition_4 = int32_t(matrix_fragment_24) >= int32_t(matrix_fragment_7);

                    packed_half_11 = packed_half_11 & 0x3u;

                    condition_0 = int32_t(matrix_accumulator_21) < int32_t(0u) || condition_0;

                    condition_1 = int32_t(matrix_accumulator_23) < int32_t(0u) || condition_1;

                    condition_5 = int32_t(matrix_accumulator_22) < int32_t(0u) || condition_5;

                    condition_6 = int32_t(matrix_accumulator_21) < int32_t(matrix_fragment_12) &&
                                  !condition_6;

                    condition_3 = int32_t(matrix_fragment_24) < int32_t(0u) || condition_3;

                    condition_4 = int32_t(matrix_accumulator_23) < int32_t(matrix_fragment_12) &&
                                  !condition_4;

                    matrix_fragment_25 =
                        uint32_t(combine_words(matrix_accumulator_18, 0u) << 0x1au >> 32) +
                        uniform_index_word_4;

                    matrix_fragment_26 = packed_half_11 + uniform_parameter_word_5;

                    condition_0 = condition_6 | condition_0 | condition_5;

                    condition_1 = condition_4 | condition_1 | condition_3;

                    condition_5 = int32_t(matrix_fragment_25) >= int32_t(matrix_accumulator_17);

                    condition_6 = int32_t(matrix_fragment_26) >= int32_t(matrix_fragment_6);

                    condition_3 = int32_t(matrix_fragment_26) >= int32_t(matrix_fragment_7);

                    condition_5 = int32_t(matrix_fragment_25) < int32_t(0u) || condition_5;

                    condition_6 = int32_t(matrix_fragment_26) < int32_t(0u) || condition_6;

                    condition_3 =
                        int32_t(matrix_fragment_25) < int32_t(matrix_fragment_12) && !condition_3;

                    condition_3 = condition_3 | condition_5 | condition_6;
                    if (!condition_2) {
                        matrix_fragment_9 = matrix_fragment_14 & 0xfu;

                        packed_half_10 = matrix_accumulator_19 * 0x4u;

                        packed_half_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(index_word_0)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_2)) *
                                    int64_t(int32_t(matrix_fragment_15)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                                combine_words(packed_half_10, packed_half_11));

                        const uint64_t byte_address_1 = scaled_word_address(
                            combine_words(uniform_index_word_14, uniform_address_word_15),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        packed_half_10 = uint32_t(byte_address_1);

                        packed_half_11 = uint32_t(byte_address_1 >> 32);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11),
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x4ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x8ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0xcull,
                                                    0u, memory_ranges, error_counts);
                    }
                    if (!condition_0) {
                        matrix_fragment_9 = matrix_accumulator_20 & 0xfu;

                        packed_half_10 = matrix_accumulator_22 * 0x4u;

                        packed_half_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(index_word_0)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_2)) *
                                    int64_t(int32_t(matrix_accumulator_21)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                                combine_words(packed_half_10, packed_half_11));

                        const uint64_t byte_address_2 = scaled_word_address(
                            combine_words(uniform_index_word_14, uniform_address_word_15),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        packed_half_10 = uint32_t(byte_address_2);

                        packed_half_11 = uint32_t(byte_address_2 >> 32);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11),
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x4ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x8ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0xcull,
                                                    0u, memory_ranges, error_counts);
                    }
                    if (!condition_1) {
                        matrix_fragment_9 = packed_half_16 & 0xfu;

                        packed_half_10 = matrix_fragment_24 * 0x4u;

                        packed_half_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(index_word_0)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_2)) *
                                    int64_t(int32_t(matrix_accumulator_23)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                                combine_words(packed_half_10, packed_half_11));

                        const uint64_t byte_address_3 = scaled_word_address(
                            combine_words(uniform_index_word_14, uniform_address_word_15),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        packed_half_10 = uint32_t(byte_address_3);

                        packed_half_11 = uint32_t(byte_address_3 >> 32);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11),
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x4ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x8ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0xcull,
                                                    0u, memory_ranges, error_counts);
                    }
                    matrix_fragment_14 = packed_half_3 + matrix_fragment_14 + packed_half_3;
                    if (!condition_3) {
                        matrix_fragment_9 = matrix_accumulator_18 & 0xfu;

                        packed_half_10 = matrix_fragment_26 * 0x4u;

                        packed_half_11 = 0u;

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            int64_t(int32_t(matrix_fragment_9)) *
                                                    int64_t(int32_t(index_word_0)) +
                                                0ull);

                        split_address_words(
                            matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_2)) *
                                    int64_t(int32_t(matrix_fragment_25)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));

                        split_address_words(matrix_fragment_8, matrix_fragment_9,
                                            combine_words(matrix_fragment_8, matrix_fragment_9) +
                                                combine_words(packed_half_10, packed_half_11));

                        const uint64_t byte_address_4 = scaled_word_address(
                            combine_words(uniform_index_word_14, uniform_address_word_15),
                            combine_words(matrix_fragment_8, matrix_fragment_9));
                        packed_half_10 = uint32_t(byte_address_4);

                        packed_half_11 = uint32_t(byte_address_4 >> 32);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11),
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x4ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0x8ull,
                                                    0u, memory_ranges, error_counts);

                        pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                        0xcull,
                                                    0u, memory_ranges, error_counts);
                    }
                    matrix_fragment_14 = packed_half_3 + matrix_fragment_14 + packed_half_3;

                    packed_half_16 = matrix_fragment_13 * 0x4u + packed_half_16;

                    matrix_accumulator_18 = matrix_fragment_13 * 0x4u + matrix_accumulator_18;

                    condition_0 = uint32_t(matrix_fragment_14) >= uint32_t(0x100u);

                    matrix_accumulator_20 = matrix_fragment_13 * 0x4u + matrix_accumulator_20;
                } while (!condition_0);
            }
        }
    }
    matrix_fragment_8 = blockIdx.z;

    matrix_fragment_9 = blockIdx.y;

    condition_0 = (matrix_fragment_8 | matrix_fragment_9 | uniform_block_index_13) != 0;

    if (condition_0) {
        return;
    }

    matrix_fragment_8 = max(int32_t(matrix_fragment_12), int32_t(matrix_fragment_5));

    packed_half_10 = min(int32_t(matrix_fragment_6), int32_t(uniform_index_word_7));

    uniform_index_word_6 = parameters[parameter_tensor_width_21];

    matrix_fragment_9 = 0u - matrix_fragment_8 + matrix_accumulator_17;

    uniform_index_word_10 = parameters[parameter_tensor_width_21];

    matrix_fragment_7 = max(int32_t(matrix_fragment_7), int32_t(packed_half_10));

    condition_0 = int32_t(matrix_fragment_9) >= int32_t(0x1u);

    uniform_index_word_4 = parameters[parameter_output_downsample_optional_low_18];
    uniform_parameter_word_5 = parameters[parameter_output_downsample_optional_high_19];

    uniform_index_word_8 = parameters[parameter_output_downsample_optional_low_18];
    uniform_address_word_9 = parameters[parameter_output_downsample_optional_high_19];
    if (condition_0) {
        matrix_fragment_9 = matrix_fragment_9 * matrix_fragment_6;

        matrix_fragment_9 = matrix_fragment_9 * 0x10u;

        condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_fragment_9);
        if (!condition_0) {
            packed_half_16 = matrix_fragment_4;
            packed_half_11 = boundary_tail_steps<true>(
                packed_half_3 + matrix_fragment_4, matrix_fragment_9, packed_half_3);

            packed_half_10 = packed_half_11 + 0x1u;

            condition_0 = uint32_t(packed_half_11) >= uint32_t(0x3u);

            matrix_fragment_14 = packed_half_10 & 0x3u;
            condition_1 = matrix_fragment_14 != 0;
            if (condition_1) {
                matrix_accumulator_18 =
                    signed_word_magnitude(matrix_fragment_6);

                condition_2 = int32_t(matrix_fragment_6) != int32_t(0u);

                matrix_fragment_14 = 0u - matrix_fragment_14;

                matrix_fragment_12 = __float_as_uint(__int2float_ru(matrix_accumulator_18));

                matrix_accumulator_17 = ~matrix_fragment_6;

                packed_half_16 = matrix_fragment_4;

                matrix_fragment_12 = reciprocal_bits(matrix_fragment_12);

                packed_half_10 = matrix_fragment_12 + 0xffffffeu;

                packed_half_11 = __float2uint_rz(__uint_as_float(packed_half_10));

                packed_half_10 = 0u;

                matrix_fragment_15 = 0u - packed_half_11;

                matrix_fragment_15 = matrix_fragment_15 * matrix_accumulator_18;

                matrix_fragment_15 = uint32_t(uint64_t(packed_half_11) * matrix_fragment_15 +
                                                  combine_words(packed_half_10, packed_half_11) >>
                                              32);
                do {
                    matrix_accumulator_21 = uint32_t(uint32_t(packed_half_16) >> 0x4u);

                    matrix_fragment_6 = uniform_index_word_6;

                    matrix_fragment_14 = matrix_fragment_14 + 0x1u;

                    matrix_fragment_12 = int32_t(matrix_accumulator_21) < 0
                                             ? 0u - matrix_accumulator_21
                                             : matrix_accumulator_21;

                    matrix_accumulator_20 =
                        signed_word_magnitude(matrix_fragment_6);

                    packed_half_10 =
                        uint32_t(uint64_t(matrix_fragment_15) * matrix_fragment_12 >> 32);

                    packed_half_11 = 0u - packed_half_10;

                    packed_half_11 = matrix_accumulator_20 * packed_half_11 + matrix_fragment_12;

                    condition_3 = uint32_t(matrix_accumulator_18) > uint32_t(packed_half_11);

                    if (!condition_3) {
                        packed_half_11 = packed_half_11 + (0u - matrix_accumulator_20);

                        packed_half_10 = packed_half_10 + 0x1u;
                    }

                    condition_1 = uint32_t(packed_half_11) >= uint32_t(matrix_accumulator_18);

                    packed_half_11 = matrix_accumulator_21 ^ matrix_fragment_6;

                    condition_3 = int32_t(packed_half_11) >= int32_t(0u);

                    packed_half_11 = packed_half_16 & 0xfu;

                    packed_half_16 = packed_half_3 + packed_half_16;

                    if (condition_1) {
                        packed_half_10 = packed_half_10 + 0x1u;
                    }

                    if (!condition_3) {
                        packed_half_10 = 0u - packed_half_10;
                    }

                    matrix_fragment_13 = !condition_2 ? matrix_accumulator_17 : packed_half_10;

                    split_address_words(
                        packed_half_10, packed_half_11,
                        int64_t(int32_t(packed_half_11)) * int64_t(int32_t(index_word_0)) + 0ull);

                    matrix_accumulator_19 = matrix_fragment_8 + matrix_fragment_13;

                    matrix_fragment_13 = 0u - matrix_fragment_13;

                    split_address_words(packed_half_10, packed_half_11,
                                        int64_t(int32_t(address_word_2)) *
                                                int64_t(int32_t(matrix_accumulator_19)) +
                                            combine_words(packed_half_10, packed_half_11));

                    matrix_fragment_13 =
                        matrix_fragment_6 * matrix_fragment_13 + matrix_accumulator_21;

                    split_address_words(packed_half_10, packed_half_11,
                                        uint64_t(uint32_t(matrix_fragment_13)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_half_10, packed_half_11));

                    const uint64_t byte_address_5 = scaled_word_address(
                        combine_words(uniform_index_word_4, uniform_parameter_word_5),
                        combine_words(packed_half_10, packed_half_11));
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
                matrix_fragment_30 =
                    signed_word_magnitude(matrix_fragment_6);

                matrix_fragment_6 = __float_as_uint(__int2float_ru(matrix_fragment_30));

                matrix_fragment_6 = reciprocal_bits(matrix_fragment_6);

                packed_half_10 = matrix_fragment_6 + 0xffffffeu;

                packed_half_11 = __float2uint_rz(__uint_as_float(packed_half_10));

                packed_half_10 = 0u;

                matrix_fragment_13 = 0u - packed_half_11;

                matrix_fragment_13 = matrix_fragment_13 * matrix_fragment_30;

                matrix_fragment_32 = uint32_t(uint64_t(packed_half_11) * matrix_fragment_13 +
                                                  combine_words(packed_half_10, packed_half_11) >>
                                              32);
                do {
                    matrix_fragment_6 = uniform_index_word_10;

                    matrix_accumulator_23 = uint32_t(uint32_t(packed_half_16) >> 0x4u);

                    matrix_fragment_24 = packed_half_3 + packed_half_16;

                    matrix_fragment_34 =
                        signed_word_magnitude(matrix_fragment_6);

                    matrix_fragment_13 = int32_t(matrix_accumulator_23) < 0
                                             ? 0u - matrix_accumulator_23
                                             : matrix_accumulator_23;

                    matrix_fragment_14 = __float_as_uint(__int2float_ru(matrix_fragment_34));

                    matrix_fragment_27 = uint32_t(uint32_t(matrix_fragment_24) >> 0x4u);

                    matrix_fragment_12 =
                        uint32_t(uint64_t(matrix_fragment_32) * matrix_fragment_13 >> 32);

                    matrix_fragment_26 = packed_half_3 + matrix_fragment_24;

                    matrix_accumulator_17 = int32_t(matrix_fragment_27) < 0
                                                ? 0u - matrix_fragment_27
                                                : matrix_fragment_27;

                    matrix_fragment_25 = ~matrix_fragment_6;

                    packed_half_10 = 0u - matrix_fragment_12;

                    matrix_fragment_28 = packed_half_3 + matrix_fragment_26;

                    matrix_fragment_29 = uint32_t(uint32_t(matrix_fragment_26) >> 0x4u);

                    matrix_fragment_13 = matrix_fragment_34 * packed_half_10 + matrix_fragment_13;

                    matrix_fragment_14 = reciprocal_bits(matrix_fragment_14);

                    packed_half_10 = 0u;

                    matrix_fragment_31 = uint32_t(uint32_t(matrix_fragment_28) >> 0x4u);

                    condition_1 = uint32_t(matrix_fragment_30) > uint32_t(matrix_fragment_13);

                    matrix_accumulator_19 = int32_t(matrix_fragment_29) < 0
                                                ? 0u - matrix_fragment_29
                                                : matrix_fragment_29;

                    matrix_accumulator_21 = int32_t(matrix_fragment_31) < 0
                                                ? 0u - matrix_fragment_31
                                                : matrix_fragment_31;

                    if (!condition_1) {
                        matrix_fragment_13 = matrix_fragment_13 + (0u - matrix_fragment_34);
                    }

                    packed_half_11 = matrix_fragment_14 + 0xffffffeu;

                    if (!condition_1) {
                        matrix_fragment_12 = matrix_fragment_12 + 0x1u;
                    }

                    condition_0 = uint32_t(matrix_fragment_13) >= uint32_t(matrix_fragment_30);

                    matrix_fragment_30 = matrix_fragment_34;

                    packed_half_11 = __float2uint_rz(__uint_as_float(packed_half_11));

                    if (condition_0) {
                        matrix_fragment_12 = matrix_fragment_12 + 0x1u;
                    }

                    condition_0 = int32_t(matrix_fragment_6) != int32_t(0u);

                    matrix_fragment_15 = 0u - packed_half_11;

                    matrix_fragment_15 = matrix_fragment_15 * matrix_fragment_34;

                    matrix_fragment_32 =
                        uint32_t(uint64_t(packed_half_11) * matrix_fragment_15 +
                                     combine_words(packed_half_10, packed_half_11) >>
                                 32);

                    packed_half_11 = matrix_accumulator_23 ^ matrix_fragment_6;

                    matrix_fragment_14 =
                        uint32_t(uint64_t(matrix_fragment_32) * matrix_accumulator_17 >> 32);

                    condition_3 = int32_t(packed_half_11) >= int32_t(0u);

                    packed_half_11 = packed_half_16 & 0xfu;

                    matrix_accumulator_18 =
                        uint32_t(uint64_t(matrix_fragment_32) * matrix_accumulator_19 >> 32);

                    packed_half_10 = 0u - matrix_fragment_14;

                    matrix_accumulator_20 =
                        uint32_t(uint64_t(matrix_fragment_32) * matrix_accumulator_21 >> 32);

                    matrix_accumulator_17 =
                        matrix_fragment_34 * packed_half_10 + matrix_accumulator_17;

                    packed_half_10 = 0u - matrix_accumulator_18;

                    matrix_accumulator_22 = 0u - matrix_accumulator_20;

                    if (!condition_3) {
                        matrix_fragment_12 = 0u - matrix_fragment_12;
                    }

                    condition_2 = uint32_t(matrix_fragment_30) > uint32_t(matrix_accumulator_17);

                    matrix_accumulator_19 =
                        matrix_fragment_34 * packed_half_10 + matrix_accumulator_19;

                    matrix_accumulator_21 =
                        matrix_fragment_34 * matrix_accumulator_22 + matrix_accumulator_21;

                    matrix_fragment_13 = !condition_0 ? matrix_fragment_25 : matrix_fragment_12;

                    matrix_fragment_12 = matrix_fragment_27 ^ matrix_fragment_6;

                    split_address_words(
                        packed_half_10, packed_half_11,
                        int64_t(int32_t(packed_half_11)) * int64_t(int32_t(index_word_0)) + 0ull);

                    condition_5 = uint32_t(matrix_fragment_30) > uint32_t(matrix_accumulator_19);

                    condition_1 = uint32_t(matrix_fragment_30) > uint32_t(matrix_accumulator_21);

                    matrix_fragment_15 = matrix_fragment_8 + matrix_fragment_13;

                    matrix_fragment_13 = 0u - matrix_fragment_13;

                    condition_3 = int32_t(matrix_fragment_12) >= int32_t(0u);

                    matrix_fragment_12 = matrix_fragment_29 ^ matrix_fragment_6;

                    if (!condition_2) {
                        matrix_accumulator_17 = matrix_accumulator_17 + (0u - matrix_fragment_34);
                    }

                    split_address_words(packed_half_10, packed_half_11,
                                        int64_t(int32_t(address_word_2)) *
                                                int64_t(int32_t(matrix_fragment_15)) +
                                            combine_words(packed_half_10, packed_half_11));

                    if (!condition_2) {
                        matrix_fragment_14 = matrix_fragment_14 + 0x1u;
                    }

                    matrix_fragment_15 = matrix_fragment_31 ^ matrix_fragment_6;

                    condition_6 = uint32_t(matrix_accumulator_17) >= uint32_t(matrix_fragment_30);

                    if (!condition_5) {
                        matrix_accumulator_19 = matrix_accumulator_19 + (0u - matrix_fragment_34);
                    }

                    if (!condition_1) {
                        matrix_accumulator_21 = matrix_accumulator_21 + (0u - matrix_fragment_34);
                    }

                    matrix_fragment_13 =
                        matrix_fragment_6 * matrix_fragment_13 + matrix_accumulator_23;

                    if (!condition_5) {
                        matrix_accumulator_18 = matrix_accumulator_18 + 0x1u;
                    }

                    condition_5 = int32_t(matrix_fragment_15) >= int32_t(0u);

                    condition_4 = uint32_t(matrix_accumulator_19) >= uint32_t(matrix_fragment_30);

                    condition_2 = uint32_t(matrix_accumulator_21) >= uint32_t(matrix_fragment_30);

                    split_address_words(packed_half_10, packed_half_11,
                                        uint64_t(uint32_t(matrix_fragment_13)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_half_10, packed_half_11));

                    if (!condition_1) {
                        matrix_accumulator_20 = matrix_accumulator_20 + 0x1u;
                    }

                    matrix_accumulator_19 = matrix_fragment_26 & 0xfu;

                    matrix_accumulator_23 = matrix_fragment_28 & 0xfu;

                    if (condition_6) {
                        matrix_fragment_14 = matrix_fragment_14 + 0x1u;
                    }

                    condition_6 = int32_t(matrix_fragment_12) >= int32_t(0u);

                    const uint64_t byte_address_6 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(packed_half_10, packed_half_11));
                    matrix_fragment_12 = uint32_t(byte_address_6);

                    if (!condition_3) {
                        matrix_fragment_14 = 0u - matrix_fragment_14;
                    }

                    if (condition_4) {
                        matrix_accumulator_18 = matrix_accumulator_18 + 0x1u;
                    }

                    matrix_fragment_13 = uint32_t(byte_address_6 >> 32);

                    if (condition_2) {
                        matrix_accumulator_20 = matrix_accumulator_20 + 0x1u;
                    }

                    packed_half_11 = matrix_fragment_24 & 0xfu;

                    matrix_fragment_15 = !condition_0 ? matrix_fragment_25 : matrix_fragment_14;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13), 0u, memory_ranges,
                        error_counts);

                    if (!condition_6) {
                        matrix_accumulator_18 = 0u - matrix_accumulator_18;
                    }

                    split_address_words(
                        packed_half_10, packed_half_11,
                        int64_t(int32_t(packed_half_11)) * int64_t(int32_t(index_word_0)) + 0ull);

                    if (!condition_5) {
                        matrix_accumulator_20 = 0u - matrix_accumulator_20;
                    }

                    matrix_accumulator_17 = 0u - matrix_fragment_15;

                    matrix_fragment_15 = matrix_fragment_8 + matrix_fragment_15;

                    matrix_accumulator_21 =
                        !condition_0 ? matrix_fragment_25 : matrix_accumulator_18;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_25 = !condition_0 ? matrix_fragment_25 : matrix_accumulator_20;

                    matrix_accumulator_17 =
                        matrix_fragment_6 * matrix_accumulator_17 + matrix_fragment_27;

                    split_address_words(packed_half_10, packed_half_11,
                                        int64_t(int32_t(address_word_2)) *
                                                int64_t(int32_t(matrix_fragment_15)) +
                                            combine_words(packed_half_10, packed_half_11));

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    matrix_fragment_27 = matrix_fragment_8 + matrix_fragment_25;

                    matrix_fragment_25 = 0u - matrix_fragment_25;

                    split_address_words(matrix_fragment_14, matrix_fragment_15,
                                        int64_t(int32_t(matrix_accumulator_19)) *
                                                int64_t(int32_t(index_word_0)) +
                                            0ull);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    split_address_words(matrix_accumulator_18, matrix_accumulator_19,
                                        int64_t(int32_t(matrix_accumulator_23)) *
                                                int64_t(int32_t(index_word_0)) +
                                            0ull);

                    matrix_accumulator_23 = matrix_fragment_8 + matrix_accumulator_21;

                    matrix_accumulator_21 = 0u - matrix_accumulator_21;

                    split_address_words(packed_half_10, packed_half_11,
                                        uint64_t(uint32_t(matrix_accumulator_17)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_half_10, packed_half_11));

                    split_address_words(packed_half_16, matrix_accumulator_17,
                                        int64_t(int32_t(address_word_2)) *
                                                int64_t(int32_t(matrix_accumulator_23)) +
                                            combine_words(matrix_fragment_14, matrix_fragment_15));

                    const uint64_t byte_address_7 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(packed_half_10, packed_half_11));
                    matrix_fragment_14 = uint32_t(byte_address_7);

                    split_address_words(
                        matrix_accumulator_18, matrix_accumulator_19,
                        int64_t(int32_t(address_word_2)) * int64_t(int32_t(matrix_fragment_27)) +
                            combine_words(matrix_accumulator_18, matrix_accumulator_19));

                    matrix_fragment_15 =
                        matrix_fragment_6 * matrix_accumulator_21 + matrix_fragment_29;

                    matrix_accumulator_21 =
                        matrix_fragment_6 * matrix_fragment_25 + matrix_fragment_31;

                    split_address_words(packed_half_16, matrix_accumulator_17,
                                        uint64_t(uint32_t(matrix_fragment_15)) *
                                                uint64_t(uint32_t(0x4u)) +
                                            combine_words(packed_half_16, matrix_accumulator_17));

                    matrix_fragment_15 = uint32_t(byte_address_7 >> 32);

                    split_address_words(
                        matrix_accumulator_18, matrix_accumulator_19,
                        uint64_t(uint32_t(matrix_accumulator_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(matrix_accumulator_18, matrix_accumulator_19));

                    const uint64_t byte_address_8 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(packed_half_16, matrix_accumulator_17));
                    packed_half_10 = uint32_t(byte_address_8);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15), 0u, memory_ranges,
                        error_counts);

                    const uint64_t byte_address_9 = scaled_word_address(
                        combine_words(uniform_index_word_8, uniform_address_word_9),
                        combine_words(matrix_accumulator_18, matrix_accumulator_19));
                    matrix_accumulator_20 = uint32_t(byte_address_9);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    packed_half_11 = uint32_t(byte_address_8 >> 32);

                    matrix_accumulator_21 = uint32_t(byte_address_9 >> 32);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    packed_half_16 = packed_half_3 + matrix_fragment_28;

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_fragment_14, matrix_fragment_15) + 0xcull, 0u,
                        memory_ranges, error_counts);

                    condition_0 = int32_t(packed_half_16) >= int32_t(matrix_fragment_9);

                    pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11), 0u,
                                                memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                    0x4ull,
                                                0u, memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                    0x8ull,
                                                0u, memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(combine_words(packed_half_10, packed_half_11) +
                                                    0xcull,
                                                0u, memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_20, matrix_accumulator_21), 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x4ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x8ull, 0u,
                        memory_ranges, error_counts);

                    pipeline_write<CheckBounds>(
                        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0xcull, 0u,
                        memory_ranges, error_counts);
                } while (!condition_0);
            }
        }
    }
    matrix_fragment_6 = 0u - matrix_fragment_7 + matrix_fragment_6;

    matrix_fragment_8 = min(int32_t(matrix_fragment_5), int32_t(matrix_fragment_6));

    condition_0 = int32_t(matrix_fragment_8) >= int32_t(0x1u);

    if (!condition_0) {
        return;
    }

    matrix_fragment_5 = matrix_fragment_5 * matrix_fragment_6;

    matrix_accumulator_19 = matrix_fragment_5 * 0x10u;

    condition_0 = int32_t(matrix_fragment_4) >= int32_t(matrix_accumulator_19);

    if (condition_0) {
        return;
    }

    // Clear the final column-padding region only. Earlier tile/edge publication
    // and its synchronization stay here; see docs/BOUNDARY_KERNELS.md.
    const BoundaryVectorLayout padding_layout{
        combine_words(parameters[parameter_output_downsample_optional_low_18],
                      parameters[parameter_output_downsample_optional_high_19]),
        index_word_0, // component stride, uint32 words
        address_word_2, // row stride, uint32 words
        matrix_fragment_7}; // column origin, signed-word bits
    const uint32_t padding_width = matrix_fragment_6;
    const uint32_t first_padding_item = matrix_fragment_4;
    const uint32_t padding_extent = matrix_accumulator_19;
    const uint32_t block_thread_count = packed_half_3;
    clear_boundary_rows<4>(padding_layout, padding_width, first_padding_item,
                           padding_extent, block_thread_count, [&](uint64_t address) {
        pipeline_write<CheckBounds>(address, 0u, memory_ranges, error_counts);
    });
}
extern "C" __global__
// Keep the vectorized operand loads spill-free; the smaller scalar-load budget
// introduced extra local-memory traffic after vectorization.
__maxnreg__(160) void phoenix_window_attention_channels_64_downsample(KernelArguments parameters) {
    run_window_attention_channels_64_downsample<false>(parameters.words, nullptr, nullptr);
}
