// DLSS5 Phoenix optimized kernel: window_attention/channels_32/upsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
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
    parameter_optional_control_9 = 9,
    parameter_publication_optional_low_14 = 14,
    parameter_publication_optional_high_15 = 15,
    parameter_input_upsample_skip_optional_low_20 = 20,
    parameter_input_upsample_skip_optional_high_21 = 21,
    parameter_tensor_height_22 = 22,
    parameter_tensor_width_23 = 23,
};

struct alignas(8) KernelArguments {
    uint32_t words[24];
};
static_assert(sizeof(KernelArguments) == 96, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_32_upsample(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t address_word_0 = 0;
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
    uint32_t packed_half_38 = 0;
    uint32_t matrix_fragment_39 = 0;
    uint32_t matrix_fragment_40 = 0;
    uint32_t matrix_fragment_41 = 0;
    uint32_t matrix_fragment_42 = 0;
    uint32_t matrix_fragment_43 = 0;
    uint32_t matrix_accumulator_44 = 0;
    uint32_t matrix_accumulator_45 = 0;
    uint32_t packed_half_46 = 0;
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
    uint32_t matrix_accumulator_68 = 0;
    uint32_t matrix_accumulator_69 = 0;
    uint32_t matrix_accumulator_70 = 0;
    uint32_t matrix_accumulator_71 = 0;
    uint32_t matrix_fragment_72 = 0;
    uint32_t matrix_fragment_73 = 0;
    uint32_t matrix_accumulator_74 = 0;
    uint32_t matrix_accumulator_75 = 0;
    uint32_t matrix_fragment_76 = 0;
    uint32_t matrix_fragment_77 = 0;
    uint32_t matrix_fragment_78 = 0;
    uint32_t matrix_fragment_79 = 0;
    uint32_t matrix_fragment_80 = 0;
    uint32_t matrix_fragment_81 = 0;
    uint32_t matrix_fragment_82 = 0;
    uint32_t matrix_fragment_83 = 0;
    uint32_t matrix_accumulator_84 = 0;
    uint32_t matrix_accumulator_85 = 0;
    uint32_t matrix_accumulator_86 = 0;
    uint32_t matrix_accumulator_87 = 0;
    uint32_t matrix_accumulator_88 = 0;
    uint32_t matrix_accumulator_89 = 0;
    uint32_t matrix_accumulator_90 = 0;
    uint32_t matrix_accumulator_91 = 0;
    uint32_t matrix_accumulator_92 = 0;
    uint32_t matrix_accumulator_93 = 0;
    uint32_t matrix_accumulator_94 = 0;
    uint32_t matrix_accumulator_95 = 0;
    uint32_t matrix_accumulator_96 = 0;
    uint32_t matrix_accumulator_97 = 0;
    uint32_t matrix_accumulator_98 = 0;
    uint32_t matrix_accumulator_99 = 0;
    uint32_t packed_half_100 = 0;
    uint32_t packed_half_101 = 0;
    uint32_t matrix_fragment_102 = 0;
    uint32_t packed_half_103 = 0;
    uint32_t matrix_fragment_104 = 0;
    uint32_t matrix_fragment_105 = 0;
    uint32_t matrix_fragment_106 = 0;
    uint32_t matrix_fragment_107 = 0;
    uint32_t packed_half_108 = 0;
    uint32_t packed_half_109 = 0;
    uint32_t packed_half_110 = 0;
    uint32_t packed_half_111 = 0;
    uint32_t packed_half_112 = 0;
    uint32_t packed_half_113 = 0;
    uint32_t packed_half_114 = 0;
    uint32_t packed_half_115 = 0;
    uint32_t packed_half_116 = 0;
    uint32_t packed_half_117 = 0;
    uint32_t packed_half_118 = 0;
    uint32_t packed_half_119 = 0;
    uint32_t packed_half_120 = 0;
    uint32_t packed_half_121 = 0;
    uint32_t packed_half_122 = 0;
    uint32_t packed_half_123 = 0;
    uint32_t packed_half_124 = 0;
    uint32_t packed_half_125 = 0;
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
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t packed_half_156 = 0;
    uint32_t packed_half_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t packed_half_159 = 0;
    uint32_t address_word_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t matrix_fragment_162 = 0;
    uint32_t matrix_fragment_163 = 0;
    uint32_t packed_half_164 = 0;
    uint32_t packed_half_165 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_address_word_6 = 0;
    uint32_t uniform_address_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_parameter_word_13 = 0;
    uint32_t uniform_parameter_word_14 = 0;
    uint32_t uniform_parameter_word_15 = 0;
    uint32_t uniform_block_index_16 = 0;
    uint32_t uniform_parameter_word_17 = 0;
    uint32_t uniform_parameter_word_18 = 0;
    uint32_t uniform_parameter_word_19 = 0;
    uint32_t uniform_reused_word_20 = 0;
    uint32_t uniform_reused_word_21 = 0;
    uint32_t uniform_index_word_22 = 0;
    uint32_t uniform_index_word_23 = 0;
    uint32_t uniform_reused_word_24 = 0;
    uint32_t uniform_packed_half_25 = 0;
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

    address_word_0 = threadIdx.x & 31u;

    uniform_index_word_8 = blockIdx.x;

    uniform_index_word_5 = parameters[parameter_optional_control_8];

    matrix_fragment_12 = 0u;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    uniform_address_word_7 = parameters[parameter_optional_control_9];

    uniform_parameter_word_13 = blockIdx.y;

    uniform_index_word_12 = parameters[parameter_tensor_width_7];

    uniform_condition_0 = true;

    uniform_reused_word_20 = 0u;
    uniform_reused_word_21 = 0u;

    matrix_fragment_25 = parameters[parameter_tensor_height_6];

    uniform_index_word_8 = (uniform_index_word_8 << 0x3u) + uniform_index_word_5;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_5));

    address_word_160 = parameters[parameter_trained_weights_low_4];
    address_word_161 = parameters[parameter_trained_weights_high_5];

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x1u >> 32) + uniform_index_word_8;

    uniform_index_word_5 = 0u - uniform_index_word_5 + uniform_index_word_12;

    packed_half_3 = address_word_0;

    uniform_address_word_7 = (uniform_parameter_word_13 << 0x3u) + uniform_address_word_7;

    uniform_index_word_10 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    matrix_fragment_6 = address_word_0;

    uniform_condition_1 = uint32_t(uniform_index_word_5) != 0u;

    packed_half_2 = uint32_t(int32_t(packed_half_3) >> 0x2u);

    uniform_address_word_6 =
        uint32_t(combine_words(uniform_address_word_7, 0u) << 0x1u >> 32) + uniform_address_word_7;

    uniform_address_word_4 = 0u;

    matrix_fragment_4 = uint32_t(combine_words(packed_half_2, 0u) << 0x2u >> 32) + packed_half_2;

    uniform_index_word_9 = uint32_t(int32_t(uniform_address_word_6) >> 0x1u);

    uniform_address_word_6 =
        uint32_t(combine_words(uniform_index_word_12, 0u) << 0x1u >> 32) + uniform_index_word_12;

    uniform_index_word_12 = uniform_index_word_12 & 0xfffffffeu;

    matrix_fragment_5 = matrix_fragment_4 & 0xfffffffcu;

    uniform_index_word_11 = uint32_t(int32_t(uniform_address_word_6) >> 0x1u);

    matrix_fragment_13 =
        uint32_t(int64_t(int32_t(matrix_fragment_6)) * (1ll << 0x1cu) >> 32) + uniform_index_word_9;

    if (uniform_condition_1) {
        uniform_index_word_5 = uniform_index_word_5 + 0xffffffffu;
    }

    matrix_fragment_8 = packed_half_2 + uniform_index_word_10 + (0u - matrix_fragment_5);

    matrix_fragment_4 = 0;
    matrix_fragment_5 = 0;

    packed_half_2 =
        uint32_t(combine_words(matrix_fragment_25, 0u) << 0x1u >> 32) + matrix_fragment_25;

    if (uniform_condition_1) {
        uniform_address_word_6 = uint32_t(int32_t(uniform_index_word_5) >> 0x1fu);
    }

    matrix_fragment_25 = matrix_fragment_25 & 0xfffffffeu;

    matrix_fragment_6 = 0;
    matrix_fragment_7 = 0;

    matrix_fragment_24 = uint32_t(int32_t(packed_half_2) >> 0x1u);

    if (uniform_condition_1) {
        uniform_address_word_6 = uint32_t(combine_words(uniform_address_word_6, 0u) << 0x3u >> 32) +
                                 uniform_index_word_5;
    }

    matrix_fragment_36 = uniform_index_word_12;

    condition_3 = int32_t(matrix_fragment_8) >= int32_t(uniform_index_word_11);

    condition_4 = uint32_t(matrix_fragment_25) != uint32_t(0x2u);

    condition_0 = int32_t(matrix_fragment_13) >= int32_t(matrix_fragment_24);

    packed_half_2 = matrix_fragment_8 * 0x4u;

    if (uniform_condition_1) {
        uniform_address_word_6 = uint32_t(int32_t(uniform_address_word_6) >> 0x3u);
    }

    condition_3 = int32_t(matrix_fragment_8) > -1 && !condition_3;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    condition_0 = int32_t(matrix_fragment_13) < int32_t(0u) || condition_0;

    matrix_fragment_27 = condition_4 ? matrix_fragment_13 : 0u;

    uniform_index_word_12 = uniform_index_word_11 << (0x2u & 31u);

    uniform_index_word_5 = uniform_index_word_9 + 0x2u;

    if (uniform_condition_1) {
        uniform_address_word_4 =
            uniform_address_word_6 * uniform_parameter_word_13 + uniform_parameter_word_13;
    }

    condition_1 = uint32_t(matrix_fragment_25) != uint32_t(0x2u) && condition_0;

    condition_0 = uint32_t(matrix_fragment_25) == uint32_t(0x2u) || !condition_0;

    condition_2 = uint32_t(matrix_fragment_36) == uint32_t(0x2u) || condition_1;

    condition_1 = uint32_t(matrix_fragment_36) == uint32_t(0x2u) && !condition_1;

    condition_0 = condition_0 & (condition_2 | condition_3);

    matrix_fragment_26 = !condition_1 ? packed_half_2 : 0u;
    do {
        matrix_fragment_13 = matrix_fragment_12 * 0x8u;

        if (condition_0) {
            matrix_fragment_15 = packed_half_3 & 0xfffffffcu;
        }

        matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_12) >> 0x4u);

        split_address_words(packed_half_2, packed_half_3,
                            uint64_t(uint32_t(matrix_fragment_13)) * uint64_t(uint32_t(0x4u)) +
                                combine_words(address_word_160, address_word_161));

        matrix_fragment_12 = 0u;

        if (condition_0) {
            matrix_fragment_13 = 0u - matrix_fragment_15 + address_word_0;

            matrix_fragment_14 = parameters[parameter_input_low_0];
            matrix_fragment_15 = parameters[parameter_input_high_1];
        }

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        address_word_0 = threadIdx.x & 31u;

        condition_6 = uint32_t(matrix_fragment_25) != uint32_t(0x2u);

        matrix_fragment_29 = matrix_fragment_18 + 0x1u;

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x2200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        packed_half_3 = address_word_0;

        matrix_fragment_20 = address_word_0;

        packed_half_118 = uint32_t(int32_t(packed_half_3) >> 0x2u);

        matrix_fragment_19 = uint32_t(int64_t(int32_t(matrix_fragment_20)) * (1ll << 0x1cu) >> 32) +
                             uniform_index_word_5;

        packed_half_2 =
            uint32_t(combine_words(packed_half_118, 0u) << 0x2u >> 32) + packed_half_118;

        condition_1 = int32_t(matrix_fragment_19) >= int32_t(matrix_fragment_24);

        matrix_fragment_21 = packed_half_2 & 0xfffffffcu;

        if (condition_0) {
            packed_half_2 = matrix_fragment_24 * matrix_fragment_18 + matrix_fragment_27;
        }

        matrix_fragment_27 = uint32_t(int64_t(int32_t(matrix_fragment_20)) * (1ll << 0x1cu) >> 32) +
                             uniform_index_word_9;

        condition_4 = int32_t(matrix_fragment_19) < int32_t(0u) || condition_1;

        matrix_fragment_30 = packed_half_118 + uniform_index_word_10 + (0u - matrix_fragment_21);

        if (condition_0) {
            matrix_fragment_13 = packed_half_2 * uniform_index_word_12 + matrix_fragment_13;
        }

        condition_3 = int32_t(matrix_fragment_27) >= int32_t(matrix_fragment_24);

        condition_1 = uint32_t(matrix_fragment_25) != uint32_t(0x2u) && condition_4;

        condition_2 = int32_t(matrix_fragment_30) >= int32_t(uniform_index_word_11);

        if (condition_0) {
            matrix_fragment_13 = matrix_fragment_26 + matrix_fragment_13;
        }

        condition_4 = uint32_t(matrix_fragment_25) == uint32_t(0x2u) || !condition_4;

        condition_3 = int32_t(matrix_fragment_27) < int32_t(0u) || condition_3;

        condition_5 = uint32_t(matrix_fragment_36) == uint32_t(0x2u) || condition_1;

        condition_2 = int32_t(matrix_fragment_30) > -1 && !condition_2;

        if (condition_0) {
            split_address_words(matrix_fragment_14, matrix_fragment_15,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_14, matrix_fragment_15));
        }

        condition_4 = condition_4 & (condition_5 | condition_2);

        if (condition_0) {
            matrix_fragment_12 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
        }

        condition_5 = uint32_t(matrix_fragment_25) != uint32_t(0x2u) && condition_3;

        condition_0 = uint32_t(matrix_fragment_25) == uint32_t(0x2u) || !condition_3;

        condition_3 = uint32_t(matrix_fragment_36) == uint32_t(0x2u) || condition_5;

        matrix_fragment_19 = condition_6 ? matrix_fragment_19 : 0u;

        condition_0 = condition_0 & (condition_3 | condition_2);

        if (condition_4) {
            matrix_fragment_22 = parameters[parameter_input_low_0];
            matrix_fragment_23 = parameters[parameter_input_high_1];

            matrix_fragment_13 = packed_half_3 & 0xfffffffcu;

            packed_half_2 = matrix_fragment_24 * matrix_fragment_18 + matrix_fragment_19;

            matrix_fragment_21 = packed_half_3 & 0xfffffffcu;

            matrix_fragment_13 = 0u - matrix_fragment_13 + address_word_0;

            matrix_fragment_15 = 0u - matrix_fragment_21 + address_word_0;

            matrix_fragment_13 = packed_half_2 * uniform_index_word_12 + matrix_fragment_13;
        }

        if (condition_0) {
            matrix_fragment_20 = parameters[parameter_input_low_0];
            matrix_fragment_21 = parameters[parameter_input_high_1];
        }

        if (condition_4) {
            packed_half_2 = matrix_fragment_24 * matrix_fragment_29 + matrix_fragment_19;
        }

        matrix_fragment_27 = condition_6 ? matrix_fragment_27 : 0u;

        if (condition_4) {
            matrix_fragment_28 = packed_half_2 * uniform_index_word_12 + matrix_fragment_15;

            matrix_fragment_18 = parameters[parameter_input_low_0];
            matrix_fragment_19 = parameters[parameter_input_high_1];
        }

        if (condition_0) {
            matrix_fragment_15 = packed_half_3 & 0xfffffffcu;

            matrix_fragment_31 = matrix_fragment_30 * 0x4u;
        }

        condition_1 = uint32_t(matrix_fragment_36) == uint32_t(0x2u) && !condition_1;

        condition_5 = uint32_t(matrix_fragment_36) == uint32_t(0x2u) && !condition_5;

        if (condition_4) {
            packed_half_2 = matrix_fragment_30 * 0x4u;
        }

        if (condition_0) {
            matrix_fragment_15 = 0u - matrix_fragment_15 + address_word_0;

            matrix_fragment_14 = matrix_fragment_24 * matrix_fragment_29 + matrix_fragment_27;
        }

        if (condition_4) {
            matrix_fragment_30 = matrix_fragment_30 * 0x4u;
        }

        if (condition_0) {
            matrix_fragment_26 = !condition_5 ? matrix_fragment_31 : 0u;
        }

        if (condition_4) {
            packed_half_2 = !condition_1 ? packed_half_2 : 0u;
        }

        if (condition_0) {
            matrix_fragment_15 = matrix_fragment_14 * uniform_index_word_12 + matrix_fragment_15;
        }

        if (condition_4) {
            matrix_fragment_29 = !condition_1 ? matrix_fragment_30 : 0u;

            matrix_fragment_13 = packed_half_2 + matrix_fragment_13;
        }

        if (condition_0) {
            matrix_fragment_15 = matrix_fragment_26 + matrix_fragment_15;
        }

        if (condition_4) {
            matrix_fragment_29 = matrix_fragment_29 + matrix_fragment_28;
        }

        if (condition_0) {
            split_address_words(matrix_fragment_20, matrix_fragment_21,
                                int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_20, matrix_fragment_21));
        }

        matrix_fragment_14 = 0;
        matrix_fragment_15 = 0;

        if (condition_4) {
            split_address_words(matrix_fragment_18, matrix_fragment_19,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_18, matrix_fragment_19));
        }

        matrix_fragment_13 = 0u;

        if (condition_0) {
            matrix_fragment_14 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_20, matrix_fragment_21), memory_ranges, error_counts);
        }

        if (condition_4) {
            split_address_words(matrix_fragment_22, matrix_fragment_23,
                                int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_22, matrix_fragment_23));

            matrix_fragment_13 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_18, matrix_fragment_19), memory_ranges, error_counts);

            matrix_fragment_15 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_22, matrix_fragment_23), memory_ranges, error_counts);
        }

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x2000ull, matrix_fragment_28,
            matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
            error_counts);

        uniform_condition_1 = uniform_condition_0;

        uniform_condition_0 = false;

        multiply_accumulate_half(matrix_fragment_6, matrix_fragment_7, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_6,
                                 matrix_fragment_7);

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_28, matrix_fragment_29, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_30, matrix_fragment_31, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                                 matrix_fragment_5);

        matrix_fragment_12 = 0x20u;
    } while (uniform_condition_1);
    uniform_index_word_10 = parameters[parameter_tensor_height_22];
    uniform_index_word_11 = parameters[parameter_tensor_width_23];

    uniform_index_word_12 = parameters[parameter_tensor_height_6];
    uniform_parameter_word_13 = parameters[parameter_tensor_width_7];

    uniform_address_word_6 = uint32_t(int32_t(uniform_address_word_7) >> 0x1fu);

    uniform_condition_5 = false;

    uniform_condition_4 = false;

    uniform_address_word_6 =
        uint32_t(combine_words(uniform_address_word_6, 0u) << 0x2u >> 32) + uniform_address_word_7;

    uniform_condition_0 = int32_t(uniform_index_word_10) > int32_t(0u);

    uniform_index_word_9 = uniform_condition_0 ? uniform_index_word_10 : uniform_index_word_12;

    uniform_condition_0 = int32_t(uniform_index_word_11) > int32_t(0u);

    uniform_index_word_5 = uniform_index_word_9 & 0xfffffffcu;

    uniform_index_word_11 = uniform_condition_0 ? uniform_index_word_11 : uniform_parameter_word_13;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_9) >> 0x1fu);

    uniform_condition_0 = true;

    uniform_condition_3 = uint32_t(uniform_index_word_5) != uint32_t(0x4u);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_index_word_12 = uint32_t(int32_t(uniform_index_word_11) >> 0x1fu);

    uniform_index_word_10 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_9;

    uniform_index_word_9 = uint32_t(int32_t(uniform_address_word_6) >> 0x2u);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_8;

    uniform_index_word_12 =
        uint32_t(combine_words(uniform_index_word_12, 0u) << 0x2u >> 32) + uniform_index_word_11;

    uniform_address_word_6 = uint32_t(int32_t(uniform_index_word_10) >> 0x2u);

    uniform_index_word_11 = uniform_index_word_11 & 0xfffffffcu;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_index_word_12 = uint32_t(int32_t(uniform_index_word_12) >> 0x2u);

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_9 + 0x1u;

        uniform_condition_2 = int32_t(uniform_index_word_9) >= int32_t(uniform_address_word_6);

        uniform_parameter_word_13 = uniform_index_word_9 + 0x1u;
    }

    uniform_condition_1 = int32_t(uniform_index_word_10) >= int32_t(uniform_index_word_12);

    if (uniform_condition_3) {
        uniform_condition_6 = int32_t(uniform_index_word_5) >= int32_t(uniform_address_word_6);

        uniform_condition_5 =
            int32_t(uniform_address_word_7) < int32_t(0u - 0x3u) || uniform_condition_2;
    }

    uniform_condition_2 = true;

    uniform_condition_1 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x4u) && !uniform_condition_1;

    if (uniform_condition_3) {
        uniform_condition_4 =
            int32_t(uniform_address_word_7) < int32_t(0u - 0x7u) || uniform_condition_6;

        uniform_condition_0 = !uniform_condition_5;
    }

    uniform_condition_6 = uint32_t(uniform_index_word_11) == uint32_t(0x4u) || uniform_condition_5;

    uniform_parameter_word_14 = uniform_index_word_10 + 0x1u;

    if (uniform_condition_3) {
        uniform_condition_2 = !uniform_condition_4;
    }

    uniform_block_index_16 = uniform_condition_4 ? uniform_index_word_10 : 0u;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_6 | uniform_condition_1);

    uniform_condition_6 = uint32_t(uniform_index_word_11) == uint32_t(0x4u) || uniform_condition_4;

    uniform_condition_4 = true;

    uniform_index_word_5 = uint32_t(uniform_condition_0);

    uniform_condition_2 = uniform_condition_2 & (uniform_condition_6 | uniform_condition_1);

    if (uniform_condition_3) {
        uniform_condition_6 = int32_t(uniform_index_word_9) >= int32_t(uniform_address_word_6);

        uniform_condition_0 = int32_t(uniform_parameter_word_13) >= int32_t(uniform_address_word_6);
    }

    uniform_condition_1 = false;

    uniform_parameter_word_13 = uniform_condition_5 ? uniform_index_word_10 : 0u;

    if (uniform_condition_3) {
        uniform_condition_1 =
            int32_t(uniform_address_word_7) < int32_t(0u - 0x3u) || uniform_condition_6;
    }

    uniform_condition_5 = int32_t(uniform_parameter_word_14) >= int32_t(uniform_index_word_12);

    uniform_condition_6 = false;

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_address_word_7) < int32_t(0u - 0x7u) || uniform_condition_0;

        uniform_condition_4 = !uniform_condition_1;
    }

    uniform_condition_5 =
        int32_t(uniform_index_word_8) > int32_t(0u - 0x8u) && !uniform_condition_5;

    uniform_condition_0 = uint32_t(uniform_index_word_11) == uint32_t(0x4u) || uniform_condition_1;

    uniform_address_word_7 = 0u;

    uniform_address_word_6 = 0u;

    uniform_index_word_8 = 0u;

    uniform_condition_4 = uniform_condition_4 & (uniform_condition_0 | uniform_condition_5);

    uniform_condition_0 = int32_t(uniform_index_word_5) != int32_t(0u);

    if (uniform_condition_3) {
        uniform_address_word_7 =
            uniform_index_word_9 * uniform_index_word_12 + uniform_index_word_12;

        uniform_address_word_6 = uniform_index_word_9 * uniform_index_word_12;
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_9 * uniform_index_word_12;

        uniform_index_word_8 = uniform_index_word_9 * uniform_index_word_12 + uniform_index_word_12;
    }

    uniform_index_word_12 = uniform_condition_6 ? uniform_parameter_word_14 : 0u;

    uniform_parameter_word_15 = uniform_condition_1 ? uniform_parameter_word_14 : 0u;

    uniform_condition_1 = true;

    if (uniform_condition_3) {
        uniform_condition_1 = !uniform_condition_6;
    }

    uniform_condition_3 = uint32_t(uniform_index_word_11) != uint32_t(0x4u);

    uniform_condition_6 = uint32_t(uniform_index_word_11) == uint32_t(0x4u) || uniform_condition_6;

    condition_0 = uniform_condition_0;

    if (!uniform_condition_2) {
        uniform_packed_half_25 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                               select_half_operand<2, false, false>(0u));
    }

    uniform_index_word_11 =
        !uniform_condition_3 ? uniform_parameter_word_13 : uniform_index_word_10;

    uniform_reused_word_24 =
        !uniform_condition_3 ? uniform_index_word_12 : uniform_parameter_word_14;

    uniform_parameter_word_15 =
        !uniform_condition_3 ? uniform_parameter_word_15 : uniform_parameter_word_14;

    if (uniform_condition_0) {
        uniform_index_word_12 = parameters[parameter_input_upsample_skip_optional_low_20];
        uniform_parameter_word_13 = parameters[parameter_input_upsample_skip_optional_high_21];
    }

    condition_2 = uniform_condition_2;

    condition_1 = uniform_condition_4;

    uniform_parameter_word_18 =
        !uniform_condition_3 ? uniform_block_index_16 : uniform_index_word_10;

    uniform_condition_1 = uniform_condition_1 & (uniform_condition_6 | uniform_condition_5);

    if (uniform_condition_2) {
        uniform_block_index_16 = parameters[parameter_input_upsample_skip_optional_low_20];
        uniform_parameter_word_17 = parameters[parameter_input_upsample_skip_optional_high_21];
    }

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_11 + uniform_index_word_5;
    }

    if (uniform_condition_4) {
        uniform_index_word_22 = uniform_parameter_word_15 + uniform_address_word_6;
    }

    if (uniform_condition_2) {
        uniform_address_word_7 = uniform_parameter_word_18 + uniform_address_word_7;
    }

    if (uniform_condition_4) {
        uniform_parameter_word_14 = parameters[parameter_input_upsample_skip_optional_low_20];
        uniform_parameter_word_15 = parameters[parameter_input_upsample_skip_optional_high_21];
    }

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_5 << (0x7u & 31u);
    }

    if (uniform_condition_2) {
        uniform_index_word_23 = uniform_address_word_7 << (0x7u & 31u);
    }

    if (!uniform_condition_0) {
        uniform_index_word_11 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                              select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_1) {
        uniform_parameter_word_18 = parameters[parameter_input_upsample_skip_optional_low_20];
        uniform_parameter_word_19 = parameters[parameter_input_upsample_skip_optional_high_21];
    }

    if (uniform_condition_0) {
        split_address_words(uniform_address_word_6, uniform_address_word_7,
                            int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_index_word_12, uniform_parameter_word_13));
    }

    if (!uniform_condition_0) {
        uniform_index_word_11 =
            permute_signed(uniform_index_word_11, uniform_index_word_11, 0x5410u);
    }

    if (uniform_condition_4) {
        uniform_index_word_22 = uniform_index_word_22 << (0x7u & 31u);
    }

    if (uniform_condition_2) {
        split_address_words(uniform_index_word_12, uniform_parameter_word_13,
                            int64_t(int32_t(uniform_index_word_23)) * int64_t(int32_t(0x4u)) +
                                combine_words(uniform_block_index_16, uniform_parameter_word_17));
    }

    if (!uniform_condition_4) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (!uniform_condition_2) {
        uniform_packed_half_25 =
            permute_signed(uniform_packed_half_25, uniform_packed_half_25, 0x5410u);
    }

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    if (!condition_0) {
        matrix_fragment_16 = uniform_index_word_11;

        matrix_fragment_17 = uniform_index_word_11;

        matrix_fragment_18 = uniform_index_word_11;

        matrix_fragment_19 = uniform_index_word_11;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_12, matrix_fragment_13));
    }

    if (uniform_condition_1) {
        uniform_index_word_8 = uniform_reused_word_24 + uniform_index_word_8;
    }

    if (uniform_condition_4) {
        split_address_words(
            uniform_address_word_6, uniform_address_word_7,
            int64_t(int32_t(uniform_index_word_22)) * int64_t(int32_t(0x4u)) +
                combine_words(uniform_parameter_word_14, uniform_parameter_word_15));
    }

    if (!uniform_condition_4) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        combine_words(uniform_index_word_12, uniform_parameter_word_13));

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    if (uniform_condition_1) {
        uniform_index_word_8 = uniform_index_word_8 << (0x7u & 31u);
    }

    condition_0 = uniform_condition_1;

    matrix_accumulator_139 = packed_half_3 & 0xfffffffcu;

    if (condition_2) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    if (!condition_2) {
        matrix_fragment_20 = uniform_packed_half_25;

        matrix_fragment_21 = uniform_packed_half_25;

        matrix_fragment_22 = uniform_packed_half_25;

        matrix_fragment_23 = uniform_packed_half_25;
    }

    split_address_words(matrix_fragment_24, matrix_fragment_25,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    if (!condition_1) {
        matrix_fragment_28 = uniform_index_word_5;

        matrix_fragment_29 = uniform_index_word_5;

        matrix_fragment_30 = uniform_index_word_5;

        matrix_fragment_31 = uniform_index_word_5;
    }

    if (!uniform_condition_1) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_1) {
        split_address_words(
            uniform_address_word_6, uniform_address_word_7,
            int64_t(int32_t(uniform_index_word_8)) * int64_t(int32_t(0x4u)) +
                combine_words(uniform_parameter_word_18, uniform_parameter_word_19));
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_14, matrix_fragment_15), matrix_fragment_20,
            matrix_fragment_21, matrix_fragment_22, matrix_fragment_23, memory_ranges,
            error_counts);
    }

    packed_half_2 = address_word_0 * 0x2u;

    if (!uniform_condition_1) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    matrix_fragment_163 = 0u - matrix_accumulator_139 + address_word_0;

    packed_half_159 = address_word_0 + 0x4u + (0u - matrix_accumulator_139);

    matrix_fragment_36 = uint32_t(uint32_t(address_word_0) >> 0x1u);

    packed_half_156 = parameters[parameter_trained_weights_low_4];
    packed_half_157 = parameters[parameter_trained_weights_high_5];

    uniform_index_word_12 = parameters[parameter_tensor_height_6];
    uniform_parameter_word_13 = parameters[parameter_tensor_width_7];

    split_address_words(matrix_fragment_14, matrix_fragment_15,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    if (!condition_0) {
        matrix_fragment_32 = uniform_index_word_5;

        matrix_fragment_33 = uniform_index_word_5;

        matrix_fragment_34 = uniform_index_word_5;

        matrix_fragment_35 = uniform_index_word_5;
    }

    matrix_accumulator_141 = address_word_0 + 0x8u + (0u - matrix_accumulator_139);

    if (condition_0) {
        split_address_words(matrix_fragment_14, matrix_fragment_15,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_14, matrix_fragment_15));
    }

    matrix_accumulator_139 = address_word_0 + 0xcu + (0u - matrix_accumulator_139);

    packed_half_156 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_156, packed_half_157) + 0x54a0ull, memory_ranges, error_counts);

    condition_3 = (address_word_0 & 0x8u) != 0;

    if (condition_1) {
        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_24, matrix_fragment_25));
    }

    uniform_index_word_11 = blockIdx.y;

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_14, matrix_fragment_15), matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);
    }

    uniform_parameter_word_14 = parameters[parameter_optional_control_8];
    uniform_parameter_word_15 = parameters[parameter_optional_control_9];

    split_address_words(matrix_accumulator_138, matrix_accumulator_139,
                        uint64_t(uint32_t(matrix_accumulator_139)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    if (condition_1) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), matrix_fragment_28,
            matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges,
            error_counts);
    }

    split_address_words(packed_half_158, packed_half_159,
                        uint64_t(uint32_t(packed_half_159)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_59 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_138, matrix_accumulator_139) + 0x2860ull, memory_ranges,
        error_counts);

    split_address_words(matrix_fragment_162, matrix_fragment_163,
                        int64_t(int32_t(matrix_fragment_163)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_67 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_158, packed_half_159) + 0x2860ull, memory_ranges, error_counts);

    matrix_fragment_79 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_162, matrix_fragment_163) + 0x2860ull, memory_ranges,
        error_counts);

    split_address_words(matrix_accumulator_140, matrix_accumulator_141,
                        uint64_t(uint32_t(matrix_accumulator_141)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_160, address_word_161));

    matrix_fragment_39 = matrix_fragment_36 & 0x4u;

    matrix_fragment_37 = packed_half_2 & 0x8u;

    matrix_fragment_78 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_162, matrix_fragment_163) + 0x2810ull, memory_ranges,
        error_counts);

    matrix_fragment_81 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_140, matrix_accumulator_141) + 0x2860ull, memory_ranges,
        error_counts);

    split_address_words(address_word_160, address_word_161,
                        int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_160, address_word_161));

    condition_1 = (address_word_0 & 0x10u) != 0;

    condition_0 = (address_word_0 & 0x4u) != 0;

    matrix_accumulator_90 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_158, packed_half_159) + 0x2810ull, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_160, address_word_161),
                                        matrix_fragment_24, matrix_fragment_25, matrix_fragment_26,
                                        matrix_fragment_27, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x200ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_fragment_36 = matrix_fragment_39 | 0x13u & address_word_0;

    packed_half_2 = matrix_fragment_37 | 0x13u & address_word_0;

    matrix_fragment_82 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_140, matrix_accumulator_141) + 0x2810ull, memory_ranges,
        error_counts);

    matrix_fragment_36 = matrix_fragment_36 | matrix_fragment_37;

    packed_half_2 = matrix_fragment_39 | packed_half_2;

    matrix_fragment_37 = 0x8u ^ matrix_fragment_36;

    packed_half_38 = 0x10u ^ packed_half_2;

    matrix_fragment_36 = 0x18u ^ matrix_fragment_36;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_57 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_55 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_56 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_58 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_61 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_60 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_7), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_63 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        packed_half_46 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_44 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_45 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_68 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_71 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_5), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_62 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_64 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_65 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_4), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_69 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(packed_half_2), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_51 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(packed_half_38), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_54 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_37), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_52 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_36), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    matrix_fragment_7 = !condition_1 ? matrix_fragment_40 : matrix_fragment_41;

    matrix_fragment_40 = !condition_1 ? matrix_fragment_41 : matrix_fragment_40;

    matrix_fragment_37 = !condition_1 ? matrix_fragment_55 : matrix_fragment_56;

    matrix_fragment_55 = !condition_1 ? matrix_fragment_56 : matrix_fragment_55;

    matrix_fragment_36 = !condition_1 ? matrix_fragment_10 : matrix_fragment_57;

    matrix_fragment_10 = !condition_1 ? matrix_fragment_57 : matrix_fragment_10;

    packed_half_38 = !condition_1 ? matrix_fragment_58 : matrix_fragment_61;

    matrix_fragment_41 = !condition_1 ? matrix_fragment_60 : matrix_fragment_63;

    packed_half_2 = !condition_1 ? matrix_fragment_43 : packed_half_46;

    matrix_fragment_57 = !condition_0 ? matrix_fragment_36 : matrix_fragment_37;

    matrix_fragment_5 = !condition_1 ? matrix_fragment_48 : matrix_fragment_49;

    matrix_fragment_58 = !condition_1 ? matrix_fragment_61 : matrix_fragment_58;

    matrix_fragment_63 = !condition_1 ? matrix_fragment_63 : matrix_fragment_60;

    matrix_fragment_37 = !condition_0 ? matrix_fragment_37 : matrix_fragment_36;

    matrix_fragment_43 = !condition_1 ? packed_half_46 : matrix_fragment_43;

    matrix_fragment_48 = !condition_1 ? matrix_fragment_49 : matrix_fragment_48;

    matrix_fragment_36 = !condition_0 ? matrix_fragment_10 : matrix_fragment_55;

    matrix_fragment_9 = !condition_1 ? matrix_accumulator_44 : matrix_accumulator_45;

    packed_half_46 = !condition_1 ? matrix_fragment_66 : matrix_accumulator_71;

    matrix_fragment_49 = !condition_1 ? matrix_accumulator_68 : matrix_fragment_73;

    matrix_fragment_10 = !condition_0 ? matrix_fragment_55 : matrix_fragment_10;

    matrix_fragment_6 = !condition_1 ? matrix_fragment_39 : matrix_fragment_42;

    matrix_accumulator_44 = !condition_1 ? matrix_accumulator_45 : matrix_accumulator_44;

    matrix_fragment_55 = !condition_0 ? packed_half_38 : matrix_fragment_41;

    matrix_fragment_39 = !condition_1 ? matrix_fragment_42 : matrix_fragment_39;

    matrix_fragment_41 = !condition_0 ? matrix_fragment_41 : packed_half_38;

    matrix_fragment_4 = !condition_1 ? matrix_fragment_47 : matrix_fragment_50;

    matrix_fragment_42 = !condition_1 ? matrix_fragment_62 : matrix_fragment_65;

    matrix_accumulator_45 = !condition_1 ? matrix_fragment_64 : matrix_accumulator_69;

    packed_half_38 = !condition_0 ? matrix_fragment_58 : matrix_fragment_63;

    matrix_fragment_8 = !condition_1 ? matrix_fragment_51 : matrix_fragment_54;

    matrix_fragment_11 = !condition_1 ? matrix_fragment_52 : matrix_fragment_53;

    matrix_fragment_63 = !condition_0 ? matrix_fragment_63 : matrix_fragment_58;

    matrix_fragment_47 = !condition_1 ? matrix_fragment_50 : matrix_fragment_47;

    matrix_fragment_51 = !condition_1 ? matrix_fragment_54 : matrix_fragment_51;

    matrix_fragment_52 = !condition_1 ? matrix_fragment_53 : matrix_fragment_52;

    matrix_fragment_62 = !condition_1 ? matrix_fragment_65 : matrix_fragment_62;

    matrix_accumulator_69 = !condition_1 ? matrix_accumulator_69 : matrix_fragment_64;

    matrix_fragment_66 = !condition_1 ? matrix_accumulator_71 : matrix_fragment_66;

    matrix_fragment_73 = !condition_1 ? matrix_fragment_73 : matrix_accumulator_68;

    matrix_fragment_83 = !condition_0 ? matrix_fragment_49 : packed_half_46;

    matrix_fragment_53 = !condition_0 ? matrix_fragment_6 : matrix_fragment_7;

    matrix_fragment_54 = !condition_0 ? matrix_fragment_43 : matrix_accumulator_44;

    matrix_fragment_6 = !condition_0 ? matrix_fragment_7 : matrix_fragment_6;

    matrix_fragment_50 = !condition_0 ? matrix_fragment_39 : matrix_fragment_40;

    matrix_fragment_43 = !condition_0 ? matrix_accumulator_44 : matrix_fragment_43;

    matrix_fragment_39 = !condition_0 ? matrix_fragment_40 : matrix_fragment_39;

    matrix_fragment_7 = !condition_0 ? matrix_fragment_4 : matrix_fragment_5;

    matrix_accumulator_44 = !condition_0 ? matrix_fragment_5 : matrix_fragment_4;

    matrix_fragment_61 = !condition_0 ? matrix_fragment_42 : matrix_accumulator_45;

    matrix_fragment_40 = !condition_0 ? packed_half_2 : matrix_fragment_9;

    matrix_fragment_5 = !condition_0 ? matrix_fragment_8 : matrix_fragment_11;

    matrix_accumulator_45 = !condition_0 ? matrix_accumulator_45 : matrix_fragment_42;

    matrix_accumulator_75 = !condition_0 ? packed_half_46 : matrix_fragment_49;

    matrix_fragment_9 = !condition_0 ? matrix_fragment_9 : packed_half_2;

    matrix_fragment_56 = !condition_0 ? matrix_fragment_47 : matrix_fragment_48;

    matrix_fragment_11 = !condition_0 ? matrix_fragment_11 : matrix_fragment_8;

    matrix_fragment_42 = !condition_0 ? matrix_fragment_62 : matrix_accumulator_69;

    packed_half_46 = !condition_0 ? matrix_fragment_66 : matrix_fragment_73;

    matrix_fragment_60 = !condition_0 ? matrix_fragment_73 : matrix_fragment_66;

    matrix_fragment_47 = !condition_0 ? matrix_fragment_48 : matrix_fragment_47;

    matrix_fragment_8 = !condition_0 ? matrix_fragment_51 : matrix_fragment_52;

    matrix_fragment_62 = !condition_0 ? matrix_accumulator_69 : matrix_fragment_62;

    matrix_fragment_52 = !condition_0 ? matrix_fragment_52 : matrix_fragment_51;

    matrix_fragment_76 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_77 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_accumulator_70 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_accumulator_71 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_accumulator_99 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_64 = unpack_fp8_bits(matrix_fragment_20 >> 16);

    matrix_fragment_65 = unpack_fp8_bits(matrix_fragment_21 >> 16);

    matrix_accumulator_84 = unpack_fp8_bits(matrix_fragment_22 >> 16);

    matrix_accumulator_85 = unpack_fp8_bits(matrix_fragment_23 >> 16);

    matrix_fragment_58 = unpack_fp8_bits(matrix_fragment_34 >> 16);

    matrix_fragment_66 = unpack_fp8_bits(matrix_fragment_32 >> 16);

    matrix_accumulator_91 = unpack_fp8_bits(matrix_fragment_28 >> 16);

    matrix_accumulator_92 = unpack_fp8_bits(matrix_fragment_29 >> 16);

    matrix_fragment_58 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_88 = unpack_fp8_bits(matrix_fragment_30 >> 16);

    matrix_fragment_83 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_138, matrix_accumulator_139) + 0x2810ull, memory_ranges,
        error_counts);

    matrix_accumulator_89 = unpack_fp8_bits(matrix_fragment_31 >> 16);

    packed_half_2 = unpack_fp8_bits(matrix_fragment_33);

    matrix_fragment_4 = unpack_fp8_bits(matrix_fragment_35);

    matrix_fragment_33 = unpack_fp8_bits(matrix_fragment_33 >> 16);

    matrix_fragment_35 = unpack_fp8_bits(matrix_fragment_35 >> 16);

    matrix_fragment_76 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_accumulator_94 = unpack_fp8_bits(matrix_fragment_32);

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_80 = unpack_fp8_bits(matrix_fragment_34);

    matrix_accumulator_91 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_92 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_70 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_accumulator_71 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_88 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_accumulator_89 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(packed_half_46));

    matrix_accumulator_99 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(matrix_fragment_67),
                                      select_half_operand<2, false, false>(matrix_fragment_52));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1000ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    matrix_fragment_59 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_59),
                                      select_half_operand<2, false, false>(matrix_fragment_60));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1200ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    matrix_accumulator_98 = unpack_fp8_bits(matrix_fragment_16);

    matrix_accumulator_93 = unpack_fp8_bits(matrix_fragment_20);

    matrix_accumulator_95 = unpack_fp8_bits(matrix_fragment_21);

    matrix_accumulator_97 = unpack_fp8_bits(matrix_fragment_28);

    matrix_accumulator_96 = unpack_fp8_bits(matrix_fragment_29);

    matrix_accumulator_98 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_68 = unpack_fp8_bits(matrix_fragment_18);

    matrix_accumulator_97 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_69 = unpack_fp8_bits(matrix_fragment_19);

    matrix_accumulator_96 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_39));

    matrix_accumulator_93 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_95 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_54));

    matrix_accumulator_94 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_79 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                      select_half_operand<2, false, false>(matrix_fragment_79),
                                      select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_69 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_69),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_36));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                                      select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_73 = unpack_fp8_bits(matrix_fragment_31);

    matrix_accumulator_86 = unpack_fp8_bits(matrix_fragment_22);

    matrix_accumulator_87 = unpack_fp8_bits(matrix_fragment_23);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_73 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_72 = unpack_fp8_bits(matrix_fragment_30);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_89),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_88),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_72 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_37));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_86 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_87 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(packed_half_38));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                      select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                      select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                      select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    packed_half_2 = 0x2b28u;

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, 0u, 0u);

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_26),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<2, false, false>(matrix_fragment_102), 0x3b283b28u);

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

    packed_half_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_27),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    matrix_fragment_36 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_74), 0x44004400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_38),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_38),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    packed_half_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(packed_half_100), 0x3b283b28u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0xc400c400u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x400ull, matrix_fragment_36,
        matrix_fragment_37, packed_half_38, matrix_fragment_39, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, 0u, 0u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_27 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_75), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_27 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    packed_half_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_27),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_27),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_107));

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x600ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_116),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_116),
        select_half_operand<2, false, false>(packed_half_117), 0x3b283b28u);

    packed_half_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(packed_half_109));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_52 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x44004400u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                          select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_53 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x44004400u);

    packed_half_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    packed_half_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                          select_half_operand<2, false, false>(packed_half_103));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_52),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0xc400c400u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0xc400c400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(packed_half_116), 0x3b283b28u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0xc400c400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(matrix_fragment_56), 0x3b283b28u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_117 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x44004400u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_54));

    packed_half_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_117), 0xc400c400u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x44004400u);

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(0u));

    packed_half_120 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x44004400u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(packed_half_117), 0x3b283b28u);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0xc400c400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x44004400u);

    matrix_accumulator_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                                         select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_122),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_accumulator_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_44),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                                          select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_107),
                                       select_half_operand<2, false, false>(0u));

    packed_half_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_122),
        select_half_operand<2, false, false>(packed_half_123), 0x3b283b28u);

    matrix_accumulator_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44),
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x3b283b28u);

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                                          select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(0u));

    packed_half_119 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                                          select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_121), 0x3b283b28u);

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(0u));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                          select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(matrix_fragment_24));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_fragment_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_25));

    packed_half_115 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(matrix_fragment_26));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_26 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                       select_half_operand<2, false, false>(matrix_fragment_27));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_106),
                                       select_half_operand<2, false, false>(0u));

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                       select_half_operand<2, false, false>(packed_half_46));

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                   select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1400ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_accumulator_75 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                      select_half_operand<2, false, false>(0u));

    packed_half_119 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_27 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                       select_half_operand<2, false, false>(packed_half_119));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_72 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_73 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_88 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_89 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_62,
                             matrix_fragment_63);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_50, matrix_fragment_51, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_24,
                             matrix_fragment_25, matrix_fragment_26, matrix_fragment_27,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_88,
                             matrix_accumulator_89);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1600ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                       select_half_operand<2, false, false>(0u));

    packed_half_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_60 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_120),
                                       select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_60,
                             matrix_fragment_61);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_64,
                             matrix_fragment_65);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_84,
                             matrix_accumulator_85);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x800ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_45 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_98),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_99),
                            select_half_operand<2, false, false>(matrix_fragment_78));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_accumulator_44,
                             matrix_accumulator_45, packed_half_46, matrix_fragment_47,
                             matrix_fragment_48, matrix_fragment_49, matrix_accumulator_94,
                             matrix_accumulator_95);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_accumulator_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_69 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_69),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             packed_half_38, matrix_fragment_39, 0u, 0u);

    matrix_accumulator_70 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_accumulator_71 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                            select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_accumulator_44,
                             matrix_accumulator_45, packed_half_46, matrix_fragment_47,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_58,
                             matrix_fragment_59);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_38, matrix_fragment_39, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_70,
                             matrix_accumulator_71);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_accumulator_44,
                             matrix_accumulator_45, packed_half_46, matrix_fragment_47,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_80,
                             matrix_fragment_81);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0xa00ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_accumulator_44,
                             matrix_accumulator_45, packed_half_46, matrix_fragment_47,
                             matrix_fragment_50, matrix_fragment_51, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_accumulator_44 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

    matrix_accumulator_45 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

    matrix_accumulator_98 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_48, matrix_fragment_49, matrix_fragment_56,
                             matrix_fragment_57);

    packed_half_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0xc400c400u);

    matrix_fragment_54 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

    matrix_fragment_55 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_accumulator_99 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_accumulator_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_98),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_accumulator_90 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_99),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_accumulator_91),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_accumulator_98 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_98),
        select_half_operand<2, false, false>(matrix_fragment_54), 0x3b283b28u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_accumulator_99 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_99),
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x3b283b28u);

    matrix_accumulator_90 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(matrix_fragment_55), 0x3b283b28u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    matrix_accumulator_91 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    packed_half_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0xc400c400u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    matrix_fragment_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0xc400c400u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(packed_half_38, matrix_fragment_39, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             packed_half_38, matrix_fragment_39, 0u, 0u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_104), 0x3b283b28u);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(matrix_fragment_106), 0x3b283b28u);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_54),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_55),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_40),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0x44004400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_110), 0x3b283b28u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0x44004400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0xc400c400u);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_111),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(packed_half_109), 0x3b283b28u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(packed_half_100));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                          select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_fragment_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_accumulator_98 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                          select_half_operand<2, false, false>(packed_half_115));

    packed_half_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_accumulator_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_36 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_116),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0xc400c400u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_116),
        select_half_operand<2, false, false>(packed_half_117), 0x3b283b28u);

    packed_half_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_36),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_fragment_107));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(packed_half_103));

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_38),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_36),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    matrix_fragment_37 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_38),
        select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

    packed_half_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

    matrix_fragment_36 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    matrix_fragment_37 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0xc400c400u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    packed_half_110 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x44004400u);

    packed_half_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_37),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_37),
        select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_37 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_106 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_106),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                          select_half_operand<2, false, false>(packed_half_110));

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                   select_half_operand<2, false, false>(0u));

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                                       select_half_operand<2, false, false>(matrix_fragment_106));

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(matrix_fragment_37), 0x3b283b28u);

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_36 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_38),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

    packed_half_116 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                                       select_half_operand<2, false, false>(0u));

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(packed_half_112));

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_38),
        select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

    packed_half_119 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x44004400u);

    packed_half_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_116), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_56,
                             matrix_fragment_57);

    packed_half_108 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x44004400u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                          select_half_operand<2, false, false>(packed_half_111));

    packed_half_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_119), 0xc400c400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_109),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_107),
                                       select_half_operand<2, false, false>(0u));

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_117),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(0u));

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(0u));

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_117),
        select_half_operand<2, false, false>(packed_half_119), 0x3b283b28u);

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_108),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x44004400u);

    packed_half_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_40 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x44004400u);

    packed_half_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

    packed_half_121 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x44004400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<2, false, false>(packed_half_109), 0x3b283b28u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_62,
                             matrix_fragment_63);

    packed_half_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                       select_half_operand<2, false, false>(0u));

    packed_half_115 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1800ull, matrix_fragment_36,
        matrix_fragment_37, packed_half_38, matrix_fragment_39, memory_ranges, error_counts);

    packed_half_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_121), 0xc400c400u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_41),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1a00ull, matrix_accumulator_44,
        matrix_accumulator_45, packed_half_46, matrix_fragment_47, memory_ranges, error_counts);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_122),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_115),
                                       select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0xe00ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    packed_half_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_120),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                       select_half_operand<2, false, false>(matrix_fragment_52));

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_122),
        select_half_operand<2, false, false>(packed_half_123), 0x3b283b28u);

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_62),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_120),
        select_half_operand<2, false, false>(packed_half_121), 0x3b283b28u);

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<2, false, false>(matrix_fragment_63), 0x3b283b28u);

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                          select_half_operand<2, false, false>(packed_half_108));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    packed_half_109 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(packed_half_109));

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_60,
                             matrix_fragment_61);

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_92 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                       select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(matrix_accumulator_92));

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_78,
                             matrix_fragment_79);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_70,
                             matrix_accumulator_71);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0xc00ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_64,
                             matrix_fragment_65);

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_80,
                             matrix_fragment_81);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_72,
                             matrix_fragment_73);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_88,
                             matrix_accumulator_89);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, 0u, 0u);

    matrix_fragment_28 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x44004400u);

    matrix_fragment_29 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

    matrix_fragment_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_66),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, 0u, 0u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_102),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_67),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<2, false, false>(matrix_fragment_52), 0x3b283b28u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_102),
        select_half_operand<2, false, false>(matrix_fragment_53), 0x3b283b28u);

    matrix_fragment_53 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0x44004400u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<2, false, false>(matrix_fragment_58), 0x3b283b28u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(packed_half_100), 0x3b283b28u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0xc400c400u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x44004400u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_104),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_104),
        select_half_operand<2, false, false>(matrix_fragment_105), 0x3b283b28u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    matrix_fragment_32 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_59),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_32), 0x3b283b28u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30), 0x44004400u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x44004400u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x44004400u);

    matrix_fragment_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0xc400c400u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0xc400c400u);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_104),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_32),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_104),
        select_half_operand<2, false, false>(matrix_fragment_105), 0x3b283b28u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

    matrix_fragment_32 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x44004400u);

    matrix_fragment_33 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(matrix_fragment_107), 0x3b283b28u);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0xc400c400u);

    matrix_fragment_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0xc400c400u);

    packed_half_103 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0x44004400u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_103), 0xc400c400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_101),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64),
                            select_half_operand<2, false, false>(matrix_fragment_66));

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105), 0xc400c400u);

    matrix_fragment_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x44004400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_101),
        select_half_operand<2, false, false>(packed_half_103), 0x3b283b28u);

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_107),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                          select_half_operand<2, false, false>(packed_half_117));

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x44004400u);

    matrix_fragment_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_26 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x44004400u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_113),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0xc400c400u);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26), 0xc400c400u);

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                            select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<2, false, false>(packed_half_114), 0x3b283b28u);

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                          select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                            select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_26),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(matrix_fragment_107), 0x3b283b28u);

    matrix_fragment_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_fragment_34 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_24 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x44004400u);

    matrix_fragment_35 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    matrix_fragment_25 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x44004400u);

    matrix_fragment_24 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24), 0xc400c400u);

    matrix_fragment_27 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x44004400u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    matrix_fragment_26 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25), 0xc400c400u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_24),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106), 0xc400c400u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27), 0xc400c400u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_24),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    matrix_fragment_25 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_24 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(matrix_fragment_25), 0x3b283b28u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(matrix_fragment_24), 0x3b283b28u);

    matrix_fragment_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28), 0x44004400u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x44004400u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106), 0xc400c400u);

    packed_half_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29), 0x44004400u);

    matrix_fragment_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                            select_half_operand<2, false, false>(packed_half_115));

    packed_half_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_114), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_27 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_26),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(matrix_fragment_106), 0x3b283b28u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_26),
        select_half_operand<2, false, false>(matrix_fragment_27), 0x3b283b28u);

    packed_half_111 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                          select_half_operand<2, false, false>(packed_half_111));

    packed_half_114 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                          select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    packed_half_109 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                            select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_26 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_111),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                          select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(0u));

    packed_half_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x44004400u);

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_fragment_30));

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(matrix_fragment_26), 0x3b283b28u);

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_109),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1c00ull, matrix_fragment_24,
        matrix_fragment_25, matrix_fragment_26, matrix_fragment_27, memory_ranges, error_counts);

    packed_half_119 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    packed_half_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_116), 0xc400c400u);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(matrix_fragment_65));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x1e00ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    packed_half_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_119), 0xc400c400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_115),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_117),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_115),
        select_half_operand<2, false, false>(packed_half_116), 0x3b283b28u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    packed_half_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_117),
        select_half_operand<2, false, false>(packed_half_119), 0x3b283b28u);

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_35 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    packed_half_112 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                          select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_107));

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0xc400c400u);

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                          select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0xc400c400u);

    packed_half_119 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                                          select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0xc400c400u);

    packed_half_110 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                          select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                       select_half_operand<2, false, false>(0u));

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                                          select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_58),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_119),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_60),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_56),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    packed_half_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_120),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_34 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<2, false, false>(matrix_fragment_59), 0x3b283b28u);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<2, false, false>(matrix_fragment_61), 0x3b283b28u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<2, false, false>(matrix_fragment_57), 0x3b283b28u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_94,
                             matrix_accumulator_95);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_120),
        select_half_operand<2, false, false>(packed_half_121), 0x3b283b28u);

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_107 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_107),
                                        select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_38, matrix_fragment_39, matrix_accumulator_74,
                             matrix_accumulator_75);

    packed_half_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_62),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_107));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(packed_half_108));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_63),
                                       select_half_operand<2, false, false>(matrix_fragment_54));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_106),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x2aa0ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x28a0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_accumulator_44, matrix_accumulator_45, matrix_accumulator_86,
                             matrix_accumulator_87);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             packed_half_38, matrix_fragment_39, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_90,
                             matrix_accumulator_91);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_38, matrix_fragment_39, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_80,
                             matrix_fragment_81);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_96,
                             matrix_accumulator_97);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_23 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x44004400u);

    packed_half_100 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             packed_half_38, matrix_fragment_39, matrix_fragment_78,
                             matrix_fragment_79);

    packed_half_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_42, matrix_fragment_43, 0u, 0u);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_44, matrix_accumulator_45, matrix_accumulator_68,
                             matrix_accumulator_69);

    matrix_fragment_22 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_90), 0x44004400u);

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0xc400c400u);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             packed_half_46, matrix_fragment_47, matrix_accumulator_70,
                             matrix_accumulator_71);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x2ea0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0xc400c400u);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_40, matrix_fragment_41, 0u, 0u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_100),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_42),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_100),
        select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

    matrix_fragment_40 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_84), 0x44004400u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

    matrix_fragment_41 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_85), 0x44004400u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_8), 0x3b283b28u);

    matrix_fragment_105 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(matrix_fragment_9), 0x3b283b28u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x2ca0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    matrix_fragment_39 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_91), 0x44004400u);

    packed_half_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_43),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_39 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<2, false, false>(packed_half_38), 0x3b283b28u);

    packed_half_38 = minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                        0x44004400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_39),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0xc400c400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_39),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_38),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_38),
        select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

    matrix_fragment_39 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x44004400u);

    packed_half_38 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_104),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_104),
        select_half_operand<2, false, false>(packed_half_38), 0x3b283b28u);

    packed_half_38 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

    matrix_fragment_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0xc400c400u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0xc400c400u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_108),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_107), 0xc400c400u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<2, false, false>(packed_half_109), 0x3b283b28u);

    packed_half_108 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x44004400u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(matrix_fragment_107), 0x3b283b28u);

    matrix_fragment_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    multiply_accumulate_half(packed_half_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106), 0xc400c400u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_111), 0x3b283b28u);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_accumulator_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_fragment_43));

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    matrix_fragment_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_108),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_106),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<2, false, false>(matrix_fragment_7), 0x3b283b28u);

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_111));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<2, false, false>(matrix_fragment_6), 0x3b283b28u);

    matrix_fragment_6 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                            select_half_operand<2, false, false>(packed_half_109));

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20), 0x44004400u);

    packed_half_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0xc400c400u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0xc400c400u);

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(packed_half_100));

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0xc400c400u);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_49),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_accumulator_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_48),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21), 0x44004400u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x44004400u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_46),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22), 0x44004400u);

    packed_half_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<2, false, false>(packed_half_100), 0x3b283b28u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23), 0x44004400u);

    packed_half_101 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x44004400u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_46),
        select_half_operand<2, false, false>(matrix_fragment_47), 0x3b283b28u);

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0xc400c400u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_19),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_102 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_101), 0xc400c400u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_46),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_19 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_50),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(packed_half_101), 0x3b283b28u);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_38), 0x44004400u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_46),
        select_half_operand<2, false, false>(matrix_fragment_49), 0x3b283b28u);

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_102),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0xc400c400u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_18 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0x44004400u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_105 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0x44004400u);

    matrix_accumulator_85 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_104 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18), 0xc400c400u);

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_102),
        select_half_operand<2, false, false>(packed_half_46), 0x3b283b28u);

    matrix_fragment_102 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39), 0x44004400u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_51),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_105 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105), 0xc400c400u);

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x44004400u);

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                            select_half_operand<2, false, false>(packed_half_103));

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_102), 0xc400c400u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_112),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_102 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_107));

    matrix_fragment_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106), 0xc400c400u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_104),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<2, false, false>(packed_half_113), 0x3b283b28u);

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_105),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_104 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_104),
        select_half_operand<2, false, false>(matrix_fragment_51), 0x3b283b28u);

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_113));

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_105),
        select_half_operand<2, false, false>(matrix_fragment_106), 0x3b283b28u);

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                            select_half_operand<2, false, false>(matrix_fragment_104));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(0u));

    packed_half_100 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                          select_half_operand<2, false, false>(packed_half_100));

    matrix_fragment_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                       select_half_operand<2, false, false>(matrix_fragment_4));

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_5 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                            select_half_operand<2, false, false>(matrix_fragment_102));

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0xc400c400u);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_4));

    matrix_fragment_19 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

    packed_half_110 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x44004400u);

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_4 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x44004400u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_103),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5), 0xc400c400u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_114),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_38 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19), 0xc400c400u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0xc400c400u);

    matrix_fragment_19 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4), 0xc400c400u);

    packed_half_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_103),
        select_half_operand<2, false, false>(matrix_fragment_18), 0x3b283b28u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                      select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_32),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<2, false, false>(packed_half_115), 0x3b283b28u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_110),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(packed_half_108));

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_107),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_103 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(packed_half_103));

    matrix_fragment_33 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_32),
        select_half_operand<2, false, false>(matrix_fragment_33), 0x3b283b28u);

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(packed_half_38),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<2, false, false>(packed_half_115));

    packed_half_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, true, false>(matrix_fragment_19),
        select_half_operand<0, false, true>(packed_half_2), 0x37283728u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<2, false, false>(packed_half_112), 0x3b283b28u);

    matrix_fragment_18 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_107),
        select_half_operand<2, false, false>(packed_half_108), 0x3b283b28u);

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                         select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_33));

    packed_half_101 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                          select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(matrix_fragment_48));

    packed_half_2 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_19),
        select_half_operand<2, false, false>(packed_half_2), 0x3b283b28u);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_18));

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                                        select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_98,
                             matrix_accumulator_99);

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(packed_half_101));

    matrix_fragment_39 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_38),
        select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_39));

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_64, matrix_fragment_65, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_106 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_106),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                      select_half_operand<2, false, false>(matrix_fragment_106));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_24, matrix_fragment_25, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_20,
                             matrix_fragment_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_72,
                             matrix_fragment_73);

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_38, matrix_fragment_39, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_82,
                             matrix_fragment_83);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_24, matrix_fragment_25, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_26, matrix_fragment_27, matrix_fragment_76,
                             matrix_fragment_77);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_26, matrix_fragment_27, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_66, matrix_fragment_67, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                      select_half_operand<2, false, false>(matrix_fragment_5));

    multiply_accumulate_half(matrix_fragment_26, matrix_fragment_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_64, matrix_fragment_65, matrix_fragment_80,
                             matrix_fragment_81);

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                      select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_24, matrix_fragment_25, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_66, matrix_fragment_67, matrix_accumulator_96,
                             matrix_accumulator_97);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_82 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_116, packed_half_117, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_fragment_76 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_38),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_77 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_79 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_58,
                             matrix_fragment_59, 0u, 0u);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_24),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_25),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_122, packed_half_123, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_26),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_27),
                                       select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(matrix_accumulator_126));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_accumulator_128),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    multiply_accumulate_half(packed_half_142, packed_half_143, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_accumulator_90));

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                                      select_half_operand<2, false, false>(packed_half_110),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(packed_half_2));

    multiply_accumulate_half(packed_half_146, packed_half_147, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                             select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                      select_half_operand<2, false, false>(packed_half_122),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_accumulator_91));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                      select_half_operand<2, false, false>(packed_half_111),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_13), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_16 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_18 = result_word;
    }

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_12));

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                        select_half_operand<2, false, false>(packed_half_117));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                      select_half_operand<2, false, false>(packed_half_123),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_accumulator_129),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(packed_half_2));

    packed_half_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                            select_half_operand<2, false, false>(matrix_accumulator_70));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                            select_half_operand<2, false, false>(matrix_accumulator_68));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                            select_half_operand<2, false, false>(matrix_accumulator_71));

    matrix_fragment_19 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_69),
                            select_half_operand<2, false, false>(matrix_accumulator_69));

    matrix_accumulator_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(packed_half_2));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_13), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_2 = result_word;
    }

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                                      select_half_operand<2, false, false>(matrix_accumulator_75),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_18 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                                      select_half_operand<2, false, false>(matrix_accumulator_88),
                                      select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_19 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                                      select_half_operand<2, false, false>(matrix_accumulator_89),
                                      select_half_operand<2, false, false>(matrix_fragment_19));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_12), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_119 = result_word;
    }

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    multiply_accumulate_half(packed_half_150, packed_half_151, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_18 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                       select_half_operand<2, false, false>(matrix_accumulator_84));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_112 = result_word;
    }

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_16));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_14), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_15 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_18), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_85 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                   select_half_operand<2, false, false>(packed_half_2));

    packed_half_119 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                     select_half_operand<2, false, false>(packed_half_119));

    multiply_accumulate_half(packed_half_124, packed_half_125, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                     select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_17 = permute_signed(packed_half_2, packed_half_2, 0x5432u);

    matrix_fragment_12 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                             select_half_operand<2, false, false>(packed_half_146));

    multiply_accumulate_half(packed_half_144, packed_half_145, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                      select_half_operand<2, false, false>(packed_half_150),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                             select_half_operand<2, false, false>(packed_half_147));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_8, matrix_fragment_9, 0u, 0u);

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                                      select_half_operand<2, false, false>(packed_half_151),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(matrix_fragment_22));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_23));

    multiply_accumulate_half(packed_half_136, matrix_accumulator_137, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_10, matrix_fragment_11, 0u, 0u);

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                      select_half_operand<2, false, false>(matrix_fragment_72),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_9 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                      select_half_operand<2, false, false>(matrix_fragment_73),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_10 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                       select_half_operand<2, false, false>(matrix_accumulator_85));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x34a0ull, matrix_fragment_16,
        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                        select_half_operand<2, false, false>(packed_half_142));

    matrix_fragment_15 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                             select_half_operand<2, false, false>(packed_half_120));

    matrix_fragment_13 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                             select_half_operand<2, false, false>(packed_half_143));

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                            select_half_operand<2, false, false>(packed_half_121));

    packed_half_2 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_124),
                                      select_half_operand<2, false, false>(packed_half_124),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                      select_half_operand<2, false, false>(packed_half_144),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_125),
                                      select_half_operand<2, false, false>(packed_half_125),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_84 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(packed_half_145),
                                      select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_accumulator_85 = permute_signed(matrix_fragment_14, matrix_fragment_14, 0x5432u);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                   select_half_operand<2, false, false>(packed_half_2));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_86 = result_word;
    }

    packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_84),
                                     select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(matrix_accumulator_130));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_95 = result_word;
    }

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(matrix_accumulator_132));

    matrix_accumulator_87 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                       select_half_operand<2, false, false>(matrix_accumulator_85));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_93 = result_word;
    }

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_13 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_15 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                      select_half_operand<2, false, false>(packed_half_136),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_94 = result_word;
    }

    matrix_accumulator_92 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    packed_half_114 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_fragment_14));

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                     select_half_operand<2, false, false>(matrix_fragment_13));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x36a0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    matrix_accumulator_84 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_accumulator_133));

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                       select_half_operand<2, false, false>(matrix_accumulator_86));

    matrix_accumulator_85 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_accumulator_137),
                                      select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                       select_half_operand<2, false, false>(matrix_accumulator_95));

    packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                                     select_half_operand<2, false, false>(packed_half_114));

    matrix_accumulator_93 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_accumulator_93));

    matrix_fragment_10 = permute_signed(matrix_fragment_11, matrix_fragment_11, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_86), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_101 = result_word;
    }

    matrix_accumulator_85 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                       select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_9 = half_to_float_bits((matrix_accumulator_92 >> 16) * 65537u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87), 0x4100410u);

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_85), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_100 = result_word;
    }

    matrix_accumulator_84 = half_to_float_bits((matrix_accumulator_92 & 65535u) * 65537u);

    matrix_accumulator_87 = rsqrt_bits(matrix_fragment_9);

    matrix_accumulator_92 = half_to_float_bits((matrix_fragment_8 & 65535u) * 65537u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_109), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_96 = result_word;
    }

    matrix_accumulator_97 = half_to_float_bits((matrix_fragment_8 >> 16) * 65537u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11), 0x4100410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x38a0ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    matrix_accumulator_94 = permute_signed(matrix_accumulator_93, matrix_accumulator_93, 0x5432u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_2), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_95 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_115), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_98 = result_word;
    }

    matrix_accumulator_94 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                       select_half_operand<2, false, false>(matrix_accumulator_94));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_114), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_99 = result_word;
    }

    matrix_fragment_106 = half_to_float_bits((packed_half_103 & 65535u) * 65537u);

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                       select_half_operand<2, false, false>(packed_half_101));

    packed_half_103 = half_to_float_bits((packed_half_103 >> 16) * 65537u);

    matrix_accumulator_92 = rsqrt_bits(matrix_accumulator_92);

    packed_half_101 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_119), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_102 = result_word;
    }

    matrix_accumulator_85 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                       select_half_operand<2, false, false>(packed_half_100));

    matrix_accumulator_97 = rsqrt_bits(matrix_accumulator_97);

    packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(packed_half_109),
                                     select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_84 = rsqrt_bits(matrix_accumulator_84);

    matrix_accumulator_96 = permute_signed(matrix_accumulator_85, matrix_accumulator_85, 0x5432u);

    matrix_accumulator_94 = rsqrt_bits(packed_half_103);

    packed_half_103 = permute_signed(matrix_accumulator_86, matrix_accumulator_86, 0x5432u);

    matrix_accumulator_86 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                       select_half_operand<2, false, false>(packed_half_103));

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_95 = half_to_float_bits((packed_half_101 & 65535u) * 65537u);

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                     select_half_operand<2, false, false>(matrix_accumulator_98));

    matrix_accumulator_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86), 0x4100410u);

    matrix_accumulator_96 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_85),
                       select_half_operand<2, false, false>(matrix_accumulator_96));

    matrix_accumulator_98 = half_to_float_bits((packed_half_101 >> 16) * 65537u);

    matrix_accumulator_93 = rsqrt_bits(matrix_fragment_106);

    matrix_accumulator_97 = pack_half(matrix_accumulator_97, matrix_accumulator_92);

    packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                     select_half_operand<2, false, false>(matrix_accumulator_99));

    packed_half_153 = half_to_float_bits((matrix_accumulator_86 & 65535u) * 65537u);

    packed_half_103 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96), 0x4100410u);

    packed_half_113 = half_to_float_bits((matrix_accumulator_86 >> 16) * 65537u);

    matrix_fragment_106 = pack_half(matrix_accumulator_87, matrix_accumulator_84);

    matrix_accumulator_95 = rsqrt_bits(matrix_accumulator_95);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x3aa0ull, matrix_accumulator_84,
        matrix_accumulator_85, matrix_accumulator_86, matrix_accumulator_87, memory_ranges,
        error_counts);

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                            select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_96 = half_to_float_bits((packed_half_103 & 65535u) * 65537u);

    packed_half_111 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                            select_half_operand<2, false, false>(matrix_accumulator_97));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_114), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_99 = result_word;
    }

    matrix_fragment_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_98 = rsqrt_bits(matrix_accumulator_98);

    packed_half_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_91),
                            select_half_operand<2, false, false>(matrix_accumulator_97));

    matrix_accumulator_97 = half_to_float_bits((packed_half_103 >> 16) * 65537u);

    packed_half_119 = add_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                     select_half_operand<2, false, false>(matrix_fragment_102));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_2), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_101 = result_word;
    }

    packed_half_153 = rsqrt_bits(packed_half_153);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_109), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        packed_half_100 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_115), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_102 = result_word;
    }

    matrix_accumulator_92 = rsqrt_bits(packed_half_113);

    matrix_accumulator_96 = rsqrt_bits(matrix_accumulator_96);

    matrix_accumulator_97 = rsqrt_bits(matrix_accumulator_97);

    matrix_accumulator_93 = pack_half(matrix_accumulator_94, matrix_accumulator_93);

    matrix_accumulator_95 = pack_half(matrix_accumulator_98, matrix_accumulator_95);

    packed_half_152 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    packed_half_153 = pack_half(matrix_accumulator_92, packed_half_153);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_110),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_90),
                            select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    packed_half_164 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_88),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    packed_half_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_70),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    packed_half_165 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    packed_half_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_89),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_71),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x3ca0ull, matrix_accumulator_88,
        matrix_accumulator_89, matrix_accumulator_90, matrix_accumulator_91, memory_ranges,
        error_counts);

    packed_half_110 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_69),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x3ea0ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    packed_half_114 = add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                     select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_accumulator_75 = pack_half(matrix_accumulator_97, matrix_accumulator_96);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x40a0ull, matrix_accumulator_96,
        matrix_accumulator_97, matrix_accumulator_98, matrix_accumulator_99, memory_ranges,
        error_counts);

    packed_half_2 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                   select_half_operand<2, false, false>(packed_half_101));

    packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(packed_half_109),
                                     select_half_operand<2, false, false>(packed_half_100));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x30a0ull, matrix_accumulator_68,
        matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                     select_half_operand<2, false, false>(matrix_fragment_102));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x42a0ull, packed_half_100,
        packed_half_101, matrix_fragment_102, packed_half_103, memory_ranges, error_counts);

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                                          select_half_operand<2, false, false>(packed_half_153));

    packed_half_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                                          select_half_operand<2, false, false>(packed_half_153));

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                                          select_half_operand<2, false, false>(packed_half_153));

    packed_half_153 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                          select_half_operand<2, false, false>(packed_half_153));

    packed_half_146 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_164),
                                       select_half_operand<2, false, false>(0u));

    packed_half_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_165),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_73 = permute_signed(packed_half_2, packed_half_2, 0x5432u);

    matrix_fragment_22 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(matrix_fragment_23));

    packed_half_151 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_23 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(matrix_fragment_72));

    packed_half_147 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    packed_half_157 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                     select_half_operand<2, false, false>(matrix_fragment_73));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x32a0ull, matrix_fragment_72,
        matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    packed_half_2 = pack_half(0u, packed_half_156);

    matrix_fragment_104 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_104),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_156 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_157), 0x4100410u);

    packed_half_152 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_157 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_152),
                                        select_half_operand<2, false, false>(packed_half_157));

    packed_half_157 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_111),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_111 = half_to_float_bits((packed_half_156 >> 16) * 65537u);

    packed_half_164 = half_to_float_bits((packed_half_156 & 65535u) * 65537u);

    matrix_fragment_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_152 = rsqrt_bits(packed_half_164);

    packed_half_156 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_157),
                                    select_half_operand<2, false, false>(0u));

    packed_half_111 = rsqrt_bits(packed_half_111);

    packed_half_108 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_108),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_157 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_106),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_106 = permute_signed(packed_half_109, packed_half_109, 0x5432u);

    matrix_fragment_105 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_105),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_108 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_108),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_107 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_107),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_109 = add_half_pairs(select_half_operand<2, false, false>(packed_half_109),
                                     select_half_operand<2, false, false>(matrix_fragment_106));

    matrix_fragment_105 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                        select_half_operand<2, false, false>(packed_half_156));

    matrix_fragment_106 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                        select_half_operand<2, false, false>(packed_half_157));

    matrix_fragment_107 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_107),
                                        select_half_operand<2, false, false>(packed_half_108));

    packed_half_111 = pack_half(packed_half_111, packed_half_152);

    packed_half_152 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0x4100410u);

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                       select_half_operand<2, false, false>(0u));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_110),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                       select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(packed_half_110));

    matrix_fragment_16 = half_to_float_bits((packed_half_152 & 65535u) * 65537u);

    matrix_fragment_17 = half_to_float_bits((packed_half_152 >> 16) * 65537u);

    packed_half_124 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_124),
                                          select_half_operand<2, false, false>(packed_half_111));

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(packed_half_111));

    matrix_fragment_16 = rsqrt_bits(matrix_fragment_16);

    packed_half_142 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                          select_half_operand<2, false, false>(packed_half_111));

    packed_half_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                          select_half_operand<2, false, false>(packed_half_111));

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_17 = rsqrt_bits(matrix_fragment_17);

    matrix_fragment_19 = permute_signed(packed_half_112, packed_half_112, 0x5432u);

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                       select_half_operand<2, false, false>(0u));

    packed_half_112 = add_half_pairs(select_half_operand<2, false, false>(packed_half_112),
                                     select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    packed_half_149 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0x4100410u);

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                    select_half_operand<2, false, false>(0u));

    packed_half_148 = pack_half(matrix_fragment_17, matrix_fragment_16);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                       select_half_operand<2, false, false>(packed_half_112));

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = permute_signed(packed_half_115, packed_half_115, 0x5432u);

    packed_half_115 = add_half_pairs(select_half_operand<2, false, false>(packed_half_115),
                                     select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    packed_half_150 = half_to_float_bits((packed_half_149 & 65535u) * 65537u);

    matrix_fragment_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0x4100410u);

    packed_half_115 = permute_signed(packed_half_114, packed_half_114, 0x5432u);

    packed_half_146 = half_to_float_bits((packed_half_149 >> 16) * 65537u);

    packed_half_149 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(packed_half_114),
                                        select_half_operand<2, false, false>(packed_half_115));

    multiply_accumulate_half(packed_half_114, packed_half_115, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_14,
                             matrix_fragment_15);

    packed_half_151 = half_to_float_bits((matrix_fragment_13 & 65535u) * 65537u);

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_124),
                                       select_half_operand<2, false, false>(packed_half_149));

    packed_half_149 = half_to_float_bits((matrix_fragment_13 >> 16) * 65537u);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_120),
                                       select_half_operand<2, false, false>(0u));

    packed_half_144 = rsqrt_bits(packed_half_151);

    packed_half_120 = permute_signed(packed_half_119, packed_half_119, 0x5432u);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x4100410u);

    packed_half_149 = rsqrt_bits(packed_half_149);

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(packed_half_119),
                                        select_half_operand<2, false, false>(packed_half_120));

    packed_half_142 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                                          select_half_operand<2, false, false>(packed_half_148));

    multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9);

    packed_half_124 = half_to_float_bits((matrix_fragment_12 & 65535u) * 65537u);

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_148));

    packed_half_125 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_125),
                                          select_half_operand<2, false, false>(packed_half_148));

    matrix_fragment_8 = half_to_float_bits((matrix_fragment_12 >> 16) * 65537u);

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(packed_half_148));

    packed_half_143 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x4100410u);

    packed_half_119 = rsqrt_bits(packed_half_124);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                       select_half_operand<2, false, false>(packed_half_143));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                       select_half_operand<2, false, false>(0u));

    packed_half_142 = half_to_float_bits((matrix_fragment_9 & 65535u) * 65537u);

    matrix_fragment_8 = rsqrt_bits(matrix_fragment_8);

    packed_half_143 = half_to_float_bits((matrix_fragment_9 >> 16) * 65537u);

    packed_half_144 = pack_half(packed_half_149, packed_half_144);

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_125),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    packed_half_147 = rsqrt_bits(packed_half_150);

    packed_half_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                          select_half_operand<2, false, false>(packed_half_144));

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                            select_half_operand<2, false, false>(packed_half_144));

    packed_half_146 = rsqrt_bits(packed_half_146);

    multiply_accumulate_half(packed_half_124, packed_half_125, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_10,
                             matrix_fragment_11);

    packed_half_142 = rsqrt_bits(packed_half_142);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(packed_half_144));

    packed_half_143 = rsqrt_bits(packed_half_143);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_144));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_132 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_half(matrix_fragment_8, packed_half_119);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    packed_half_119 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_162, matrix_fragment_163) + 0x58b0ull, memory_ranges,
        error_counts);

    matrix_fragment_11 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                      select_half_operand<2, false, false>(matrix_accumulator_132));

    matrix_accumulator_137 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_133),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_146 = pack_half(packed_half_146, packed_half_147);

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_142 = pack_half(packed_half_143, packed_half_142);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(packed_half_142));

    matrix_accumulator_85 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                      select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_142));

    matrix_accumulator_84 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(matrix_accumulator_84));

    packed_half_122 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                          select_half_operand<2, false, false>(packed_half_146));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                          select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<2, false, false>(packed_half_142));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                          select_half_operand<2, false, false>(packed_half_142));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<0, false, false>(packed_half_2));

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_accumulator_84, matrix_accumulator_85, matrix_accumulator_86,
                             matrix_accumulator_87);

    packed_half_122 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_123 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_104 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                                        select_half_operand<2, false, false>(matrix_fragment_105));

    matrix_fragment_106 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                        select_half_operand<2, false, false>(matrix_fragment_107));

    matrix_fragment_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_123),
                      select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_fragment_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                      select_half_operand<2, false, false>(matrix_accumulator_128));

    multiply_accumulate_half(packed_half_116, packed_half_117, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_22, matrix_fragment_23, matrix_accumulator_88,
                             matrix_accumulator_89);

    matrix_accumulator_88 = 0x29c0u;

    multiply_accumulate_half(packed_half_122, packed_half_123, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_20, matrix_fragment_21, matrix_accumulator_90,
                             matrix_accumulator_91);

    packed_half_2 = permute_signed(packed_half_2, matrix_accumulator_88, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_92,
                             matrix_accumulator_93);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_16, matrix_fragment_17, matrix_accumulator_94,
                             matrix_accumulator_95);

    packed_half_109 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_109),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_111),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_96,
                             matrix_accumulator_97);

    packed_half_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_113),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_115),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_98,
                             matrix_accumulator_99);

    packed_half_108 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_108),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_110),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_112),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_fragment_10, matrix_fragment_11, packed_half_100,
                             packed_half_101);

    packed_half_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_114),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_109 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0x3c203c20u);

    packed_half_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_121),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_106, matrix_fragment_107, matrix_fragment_104,
                             matrix_fragment_105, matrix_fragment_106, matrix_fragment_107,
                             matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_102,
                             packed_half_103);

    packed_half_111 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0x3c203c20u);

    packed_half_125 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_125),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_113 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0x3c203c20u);

    packed_half_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_120),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_115 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0x3c203c20u);

    packed_half_124 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_124),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_108 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0x3c203c20u);

    packed_half_110 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0x3c203c20u);

    packed_half_112 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0x3c203c20u);

    packed_half_114 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_114), 0x3c203c20u);

    matrix_fragment_8 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_8),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_109 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_109), 0x3e473e47u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_111 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_111), 0x3e473e47u);

    packed_half_113 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_113), 0x3e473e47u);

    packed_half_115 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_115), 0x3e473e47u);

    packed_half_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_116),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_121), 0x3c203c20u);

    packed_half_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_122),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_125 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_125), 0x3c203c20u);

    matrix_accumulator_132 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_108 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_108), 0x3e473e47u);

    matrix_accumulator_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_110 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_110), 0x3e473e47u);

    packed_half_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_117),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_112 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_112), 0x3e473e47u);

    packed_half_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_123),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_114 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_114), 0x3e473e47u);

    matrix_accumulator_133 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_120), 0x3c203c20u);

    matrix_accumulator_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_124 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_124), 0x3c203c20u);

    matrix_accumulator_96 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_121 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_121), 0x3e473e47u);

    matrix_accumulator_92 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_125 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_125), 0x3e473e47u);

    matrix_accumulator_137 = (packed_half_109 << 0x5u) + 0x7ff88000u;

    packed_half_136 = (packed_half_111 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_131 = (packed_half_113 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_130 = (packed_half_115 << 0x5u) + 0x7ff88000u;

    packed_half_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_120), 0x3e473e47u);

    packed_half_124 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_124), 0x3e473e47u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3c203c20u);

    matrix_accumulator_86 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3c203c20u);

    matrix_fragment_9 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_9),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_129 = (packed_half_108 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_128 = (packed_half_110 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_126 = (packed_half_112 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_127 = (packed_half_114 << 0x5u) + 0x7ff88000u;

    packed_half_116 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_116), 0x3c203c20u);

    packed_half_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_122), 0x3c203c20u);

    matrix_accumulator_132 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x3c203c20u);

    matrix_accumulator_134 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_68, matrix_accumulator_69, 0u, 0u);

    packed_half_117 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_117), 0x3c203c20u);

    matrix_accumulator_97 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_123), 0x3c203c20u);

    matrix_accumulator_93 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_133 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133), 0x3c203c20u);

    matrix_accumulator_135 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3c203c20u);

    matrix_accumulator_94 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_143 = (packed_half_121 << 0x5u) + 0x7ff88000u;

    matrix_fragment_106 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_106),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_142 = (packed_half_125 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x3e473e47u);

    matrix_accumulator_86 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_70, matrix_accumulator_71, 0u, 0u);

    packed_half_147 = (packed_half_120 << 0x5u) + 0x7ff88000u;

    packed_half_100 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                                     select_half_operand<2, false, false>(matrix_accumulator_128));

    packed_half_146 = (packed_half_124 << 0x5u) + 0x7ff88000u;

    packed_half_101 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                                     select_half_operand<2, false, false>(matrix_accumulator_127));

    packed_half_116 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_116), 0x3e473e47u);

    packed_half_122 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_122), 0x3e473e47u);

    matrix_accumulator_132 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_132), 0x3e473e47u);

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3e473e47u);

    matrix_accumulator_96 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3c203c20u);

    multiply_accumulate_half(matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_accumulator_92 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3c203c20u);

    matrix_accumulator_95 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_117 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_117), 0x3e473e47u);

    matrix_fragment_107 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_107),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_123 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_123), 0x3e473e47u);

    matrix_accumulator_133 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_133), 0x3e473e47u);

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3e473e47u);

    matrix_fragment_9 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_102, packed_half_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_accumulator_87 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3c203c20u);

    matrix_accumulator_97 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3c203c20u);

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                                       select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_6 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                       select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_accumulator_93 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3c203c20u);

    packed_half_100 = add_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                     select_half_operand<2, false, false>(packed_half_101));

    packed_half_144 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_145 = (matrix_accumulator_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                       select_half_operand<2, false, false>(packed_half_142));

    matrix_accumulator_96 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_96), 0x3e473e47u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                       select_half_operand<2, false, false>(packed_half_146));

    matrix_accumulator_92 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_92), 0x3e473e47u);

    packed_half_157 = (packed_half_116 << 0x5u) + 0x7ff88000u;

    packed_half_155 = (packed_half_122 << 0x5u) + 0x7ff88000u;

    packed_half_153 = (matrix_accumulator_132 << 0x5u) + 0x7ff88000u;

    packed_half_151 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_94 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x3c203c20u);

    matrix_fragment_106 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106), 0x3c203c20u);

    packed_half_156 = (packed_half_117 << 0x5u) + 0x7ff88000u;

    packed_half_154 = (packed_half_123 << 0x5u) + 0x7ff88000u;

    packed_half_152 = (matrix_accumulator_133 << 0x5u) + 0x7ff88000u;

    packed_half_150 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    matrix_fragment_9 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9), 0x3e473e47u);

    matrix_accumulator_87 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3e473e47u);

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    matrix_accumulator_97 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_97), 0x3e473e47u);

    matrix_accumulator_93 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_93), 0x3e473e47u);

    matrix_accumulator_95 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3c203c20u);

    matrix_fragment_107 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_107), 0x3c203c20u);

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(packed_half_100),
                                       select_half_operand<2, false, false>(matrix_fragment_5));

    matrix_accumulator_94 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_94), 0x3e473e47u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(packed_half_145));

    matrix_fragment_106 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_106), 0x3e473e47u);

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                                       select_half_operand<2, false, false>(packed_half_155));

    packed_half_125 = (matrix_accumulator_96 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                                       select_half_operand<2, false, false>(packed_half_151));

    packed_half_123 = (matrix_accumulator_92 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_86 = add_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                                           select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_95 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_95), 0x3e473e47u);

    matrix_fragment_104 = add_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                                         select_half_operand<2, false, false>(packed_half_150));

    matrix_fragment_107 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_107), 0x3e473e47u);

    packed_half_149 = (matrix_fragment_9 << 0x5u) + 0x7ff88000u;

    packed_half_148 = (matrix_accumulator_87 << 0x5u) + 0x7ff88000u;

    matrix_fragment_5 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_5),
                                       select_half_operand<2, false, false>(matrix_fragment_6));

    packed_half_120 = (matrix_accumulator_97 << 0x5u) + 0x7ff88000u;

    packed_half_116 = (matrix_accumulator_93 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    packed_half_121 = (matrix_accumulator_94 << 0x5u) + 0x7ff88000u;

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(packed_half_125),
                                       select_half_operand<2, false, false>(packed_half_123));

    packed_half_124 = (matrix_fragment_106 << 0x5u) + 0x7ff88000u;

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                                       select_half_operand<2, false, false>(matrix_fragment_104));

    multiply_accumulate_half(packed_half_114, packed_half_115, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_68, matrix_accumulator_69, 0u, 0u);

    packed_half_117 = (matrix_accumulator_95 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(packed_half_148));

    packed_half_122 = (matrix_fragment_107 << 0x5u) + 0x7ff88000u;

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                                       select_half_operand<2, false, false>(packed_half_124));

    matrix_fragment_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_4),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                       select_half_operand<2, false, false>(packed_half_122));

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_70, matrix_accumulator_71, 0u, 0u);

    matrix_fragment_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_6),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = address_word_0 * 0x4u;

    condition_0 = (address_word_0 & 0x1u) != 0;

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    condition_2 = (address_word_0 & 0x2u) != 0;

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                        select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_7 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_7),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_81 = matrix_fragment_9 & 0x1cu;

    matrix_fragment_8 = condition_0 ? matrix_fragment_4 : matrix_fragment_5;

    matrix_fragment_9 = condition_0 ? matrix_fragment_7 : matrix_fragment_6;

    matrix_fragment_5 = condition_0 ? matrix_fragment_5 : matrix_fragment_4;

    matrix_fragment_6 = condition_0 ? matrix_fragment_6 : matrix_fragment_7;

    matrix_fragment_4 =
        uint32_t(int64_t(int32_t(address_word_0)) * (1ll << 0x1du) >> 32) + matrix_fragment_81;

    matrix_fragment_81 = !condition_2 ? matrix_fragment_8 : matrix_fragment_9;

    matrix_fragment_9 = !condition_2 ? matrix_fragment_9 : matrix_fragment_8;

    matrix_fragment_8 = !condition_2 ? matrix_fragment_5 : matrix_fragment_6;

    matrix_fragment_80 = !condition_2 ? matrix_fragment_6 : matrix_fragment_5;

    matrix_fragment_5 = 0x1u ^ matrix_fragment_4;

    matrix_fragment_6 = 0x2u ^ matrix_fragment_4;

    matrix_fragment_7 = 0x3u ^ matrix_fragment_4;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_81), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_8 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_80), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_80 = result_word;
    }

    matrix_fragment_82 = !condition_3 ? matrix_fragment_81 : matrix_fragment_8;

    matrix_fragment_83 = !condition_3 ? matrix_fragment_8 : matrix_fragment_81;

    matrix_fragment_107 = !condition_3 ? matrix_fragment_9 : matrix_fragment_80;

    matrix_fragment_106 = !condition_3 ? matrix_fragment_80 : matrix_fragment_9;

    matrix_fragment_81 = !condition_1 ? matrix_fragment_82 : matrix_fragment_107;

    matrix_fragment_8 = !condition_1 ? matrix_fragment_83 : matrix_fragment_106;

    matrix_fragment_82 = !condition_1 ? matrix_fragment_107 : matrix_fragment_82;

    matrix_fragment_83 = !condition_1 ? matrix_fragment_106 : matrix_fragment_83;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_8 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_8),
                                       select_half_operand<1, false, false>(matrix_fragment_8));

    matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_8, 0x5410u);

    matrix_fragment_8 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8), 0x4100410u);

    matrix_fragment_80 = half_to_float_bits((matrix_fragment_8 & 65535u) * 65537u);

    matrix_fragment_81 = half_to_float_bits((matrix_fragment_8 >> 16) * 65537u);

    matrix_fragment_80 = reciprocal_bits(matrix_fragment_80);

    matrix_fragment_9 = uint32_t(int32_t(packed_half_3) >> 0x1fu);

    matrix_fragment_81 = reciprocal_bits(matrix_fragment_81);

    matrix_fragment_9 =
        uint32_t(combine_words(matrix_fragment_9, 0u) << 0x5u >> 32) + packed_half_118;

    matrix_accumulator_99 = transpose_half_tile(matrix_accumulator_99);

    matrix_fragment_9 = matrix_fragment_9 & 0xffffffe0u;

    matrix_accumulator_98 = transpose_half_tile(matrix_accumulator_98);

    matrix_fragment_9 = packed_half_118 + (0u - matrix_fragment_9);

    multiply_accumulate_half(matrix_accumulator_96, matrix_accumulator_97, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    packed_half_114 = transpose_half_tile(packed_half_114);

    matrix_accumulator_134 = pack_half(matrix_fragment_81, matrix_fragment_80);

    matrix_fragment_80 = transpose_half_tile(packed_half_115);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    matrix_fragment_8 = packed_half_118 + 0x8u;

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_accumulator_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                      select_half_operand<2, false, false>(0u));

    packed_half_118 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_158, packed_half_159) + 0x58b0ull, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_68, matrix_accumulator_69, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_106, matrix_fragment_107, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_70, matrix_accumulator_71, 0u, 0u);

    multiply_accumulate_half(packed_half_108, packed_half_109, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(packed_half_110, packed_half_111, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_76 = result_word;
    }

    matrix_accumulator_91 = transpose_half_tile(matrix_accumulator_91);

    matrix_fragment_77 = uint32_t(int32_t(matrix_fragment_8) >> 0x1fu);

    matrix_accumulator_90 = transpose_half_tile(matrix_accumulator_90);

    matrix_fragment_77 =
        uint32_t(combine_words(matrix_fragment_77, 0u) << 0x5u >> 32) + matrix_fragment_8;

    packed_half_115 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_140, matrix_accumulator_141) + 0x58b0ull, memory_ranges,
        error_counts);

    matrix_accumulator_98 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                      select_half_operand<2, false, false>(matrix_accumulator_99));

    matrix_fragment_77 = matrix_fragment_77 & 0xffffffe0u;

    matrix_accumulator_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_81 = matrix_fragment_8 + (0u - matrix_fragment_77);

    matrix_accumulator_99 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                      select_half_operand<2, false, false>(matrix_accumulator_99));

    packed_half_114 = transpose_half_tile(packed_half_113);

    packed_half_159 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5410u);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x54b0ull, matrix_fragment_76,
        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79, memory_ranges, error_counts);

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_128),
                            select_half_operand<2, false, false>(packed_half_159));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_126),
                            select_half_operand<2, false, false>(packed_half_159));

    matrix_accumulator_91 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_91),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_129),
                            select_half_operand<2, false, false>(packed_half_159));

    matrix_fragment_83 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_135 = transpose_half_tile(packed_half_112);

    matrix_accumulator_128 = transpose_half_tile(packed_half_100);

    matrix_accumulator_126 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_114),
                                           select_half_operand<2, false, false>(0u));

    packed_half_114 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_138, matrix_accumulator_139) + 0x58b0ull, memory_ranges,
        error_counts);

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_90),
                                    select_half_operand<2, false, false>(matrix_accumulator_91));

    packed_half_112 = transpose_half_tile(matrix_accumulator_89);

    matrix_accumulator_129 = transpose_half_tile(matrix_accumulator_88);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x56b0ull, matrix_accumulator_88,
        matrix_accumulator_89, matrix_accumulator_90, matrix_accumulator_91, memory_ranges,
        error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(matrix_fragment_81), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    packed_half_113 = transpose_half_tile(packed_half_101);

    matrix_fragment_105 = transpose_half_tile(matrix_fragment_105);

    packed_half_158 = permute_signed(matrix_fragment_81, matrix_fragment_81, 0x5410u);

    packed_half_136 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_136),
                                          select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(packed_half_158));

    packed_half_136 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(packed_half_136));

    packed_half_136 = transpose_half_tile(packed_half_103);

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                    select_half_operand<2, false, false>(matrix_accumulator_126));

    matrix_accumulator_135 = transpose_half_tile(matrix_fragment_102);

    matrix_accumulator_137 = transpose_half_tile(matrix_fragment_104);

    matrix_accumulator_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_127),
                            select_half_operand<2, false, false>(packed_half_159));

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_113),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_130),
                            select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_80 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(matrix_fragment_83));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_131),
                            select_half_operand<2, false, false>(packed_half_158));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                    select_half_operand<2, false, false>(packed_half_113));

    matrix_accumulator_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_130 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                      select_half_operand<2, false, false>(0u));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_112),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_128 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                      select_half_operand<2, false, false>(0u));

    packed_half_136 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_82 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                      select_half_operand<2, false, false>(matrix_accumulator_127));

    matrix_fragment_83 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                      select_half_operand<2, false, false>(matrix_accumulator_130));

    matrix_fragment_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(packed_half_112));

    matrix_fragment_104 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(packed_half_136));

    matrix_fragment_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                      select_half_operand<2, false, false>(matrix_accumulator_128));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             packed_half_100, packed_half_101, 0u, 0u);

    multiply_accumulate_half(packed_half_112, packed_half_113, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_102, packed_half_103, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_82 = uint32_t(int64_t(int32_t(packed_half_3)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_80 = uint32_t(int64_t(int32_t(packed_half_3)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_83 = uint32_t(int32_t(matrix_fragment_82) >> 0x1fu);

    matrix_fragment_81 = uint32_t(int32_t(matrix_fragment_80) >> 0x1fu);

    matrix_fragment_83 =
        uint32_t(combine_words(matrix_fragment_83, 0u) << 0x5u >> 32) + matrix_fragment_82;

    matrix_fragment_81 =
        uint32_t(combine_words(matrix_fragment_81, 0u) << 0x5u >> 32) + matrix_fragment_80;

    matrix_fragment_83 = matrix_fragment_83 & 0xffffffe0u;

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_68, matrix_accumulator_69, 0u, 0u);

    matrix_fragment_81 = matrix_fragment_81 & 0xffffffe0u;

    packed_half_164 = matrix_fragment_82 + (0u - matrix_fragment_83);

    packed_half_165 = matrix_fragment_80 + (0u - matrix_fragment_81);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(packed_half_164), "r"(0x1fu), "r"(__activemask()));
        packed_half_164 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_134), "r"(packed_half_165), "r"(0x1fu), "r"(__activemask()));
        packed_half_165 = result_word;
    }

    multiply_accumulate_half(packed_half_136, matrix_accumulator_137, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_accumulator_132 = transpose_half_tile(matrix_accumulator_132);

    matrix_accumulator_139 = transpose_half_tile(matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_accumulator_74 = transpose_half_tile(matrix_accumulator_133);

    matrix_fragment_163 = permute_signed(packed_half_164, packed_half_164, 0x5410u);

    matrix_accumulator_138 = transpose_half_tile(matrix_accumulator_138);

    matrix_fragment_162 = permute_signed(packed_half_165, packed_half_165, 0x5410u);

    packed_half_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_150),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    packed_half_154 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_154),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    packed_half_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_155),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    packed_half_151 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_151),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    packed_half_150 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_58, matrix_fragment_59, 0u, 0u);

    packed_half_155 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                    select_half_operand<2, false, false>(0u));

    packed_half_154 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                    select_half_operand<2, false, false>(0u));

    packed_half_151 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_151),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_152),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_153),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    multiply_accumulate_half(matrix_accumulator_140, matrix_accumulator_141, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_70, matrix_accumulator_71, 0u, 0u);

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(packed_half_150));

    packed_half_150 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_142),
                                          select_half_operand<2, false, false>(packed_half_158));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(packed_half_151));

    packed_half_143 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_143),
                                          select_half_operand<2, false, false>(packed_half_158));

    matrix_accumulator_75 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                      select_half_operand<2, false, false>(0u));

    packed_half_142 = transpose_half_tile(matrix_fragment_107);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_56, matrix_fragment_57, 0u, 0u);

    matrix_fragment_56 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_157),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_156),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                       select_half_operand<2, false, false>(packed_half_155));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_133 = transpose_half_tile(matrix_fragment_106);

    matrix_accumulator_139 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                      select_half_operand<2, false, false>(0u));

    packed_half_150 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_150),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                      select_half_operand<2, false, false>(matrix_accumulator_139));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    matrix_fragment_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_138 = transpose_half_tile(packed_half_109);

    matrix_accumulator_132 = transpose_half_tile(packed_half_108);

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_100, packed_half_101, 0u, 0u);

    packed_half_136 = transpose_half_tile(packed_half_136);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_102, packed_half_103, 0u, 0u);

    matrix_accumulator_141 = transpose_half_tile(matrix_accumulator_141);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                       select_half_operand<2, false, false>(packed_half_150));

    packed_half_143 = transpose_half_tile(matrix_accumulator_137);

    matrix_accumulator_139 = transpose_half_tile(packed_half_111);

    matrix_accumulator_135 = transpose_half_tile(matrix_accumulator_135);

    packed_half_146 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_146),
                                          select_half_operand<2, false, false>(packed_half_159));

    matrix_accumulator_140 = transpose_half_tile(matrix_accumulator_140);

    packed_half_145 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_145),
                                          select_half_operand<2, false, false>(packed_half_159));

    matrix_accumulator_137 = transpose_half_tile(packed_half_110);

    packed_half_148 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_148),
                                          select_half_operand<2, false, false>(packed_half_158));

    matrix_accumulator_134 = transpose_half_tile(matrix_accumulator_134);

    packed_half_142 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_142),
                                    select_half_operand<2, false, false>(0u));

    packed_half_147 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_147),
                                          select_half_operand<2, false, false>(packed_half_159));

    packed_half_146 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_146),
                                    select_half_operand<2, false, false>(0u));

    packed_half_144 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_144),
                                          select_half_operand<2, false, false>(packed_half_159));

    packed_half_108 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                    select_half_operand<2, false, false>(packed_half_142));

    packed_half_149 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_149),
                                          select_half_operand<2, false, false>(packed_half_158));

    packed_half_145 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_145),
                                    select_half_operand<2, false, false>(0u));

    packed_half_148 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_148),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                      select_half_operand<2, false, false>(0u));

    packed_half_143 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_143),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_147),
                                       select_half_operand<2, false, false>(packed_half_146));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_144),
                                       select_half_operand<2, false, false>(packed_half_145));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_149),
                                       select_half_operand<2, false, false>(packed_half_148));

    packed_half_110 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                    select_half_operand<2, false, false>(matrix_accumulator_133));

    packed_half_111 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_136),
                                    select_half_operand<2, false, false>(packed_half_143));

    matrix_accumulator_141 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_141),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_110, packed_half_111, packed_half_112, packed_half_113);

    packed_half_109 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_140),
                                    select_half_operand<2, false, false>(matrix_accumulator_141));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                    select_half_operand<2, false, false>(0u));

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                                    select_half_operand<2, false, false>(0u));

    packed_half_112 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_137),
                                    select_half_operand<2, false, false>(packed_half_112));

    packed_half_113 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                    select_half_operand<2, false, false>(packed_half_113));

    packed_half_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_123),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    packed_half_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_124),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    packed_half_116 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    packed_half_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_122),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_106, matrix_fragment_107, matrix_accumulator_130,
                             matrix_accumulator_131);

    packed_half_125 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_125),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    packed_half_116 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                    select_half_operand<2, false, false>(0u));

    packed_half_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                            select_half_operand<2, false, false>(matrix_fragment_162));

    packed_half_122 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_122),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_126, matrix_accumulator_127, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_108, packed_half_109, matrix_accumulator_126,
                             matrix_accumulator_127);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_112, packed_half_113, matrix_accumulator_128,
                             matrix_accumulator_129);

    matrix_fragment_57 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_123),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                            select_half_operand<2, false, false>(matrix_fragment_163));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_124),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_125),
                                       select_half_operand<2, false, false>(matrix_fragment_56));

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                       select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_121),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(packed_half_122));

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_108, packed_half_109, matrix_accumulator_68,
                             matrix_accumulator_69);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_112, packed_half_113, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_106, matrix_fragment_107, matrix_accumulator_70,
                             matrix_accumulator_71);

    multiply_accumulate_half(matrix_fragment_72, matrix_fragment_73, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             packed_half_110, packed_half_111, matrix_fragment_72,
                             matrix_fragment_73);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_126),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_127),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(matrix_fragment_57));

    multiply_accumulate_half(packed_half_116, packed_half_117, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_60, matrix_fragment_61, 0u, 0u);

    matrix_fragment_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                                       select_half_operand<2, false, false>(matrix_fragment_60));

    multiply_accumulate_half(packed_half_120, packed_half_121, matrix_fragment_64,
                             matrix_fragment_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_62, matrix_fragment_63, 0u, 0u);

    matrix_fragment_60 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                                       select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_61 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                                       select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_62 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_72),
                                       select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_63 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_73),
                                       select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(packed_half_118));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(packed_half_118));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_42 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                             select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_115));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_78, matrix_fragment_79, matrix_fragment_40,
                             matrix_fragment_41);

    packed_half_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_92));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_accumulator_93));

    packed_half_46 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(matrix_accumulator_86),
                                      select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_47 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                                      select_half_operand<2, false, false>(matrix_accumulator_87),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_fragment_78, matrix_fragment_79, matrix_accumulator_44,
                             matrix_accumulator_45);

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_44 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_accumulator_96),
                                      select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_45 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_45));

    packed_half_46 = add_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                    select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_fragment_47 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                       select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_66 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                             select_half_operand<2, false, false>(packed_half_116));

    matrix_fragment_67 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                             select_half_operand<2, false, false>(packed_half_120));

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                            select_half_operand<2, false, false>(packed_half_117));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                            select_half_operand<2, false, false>(packed_half_121));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(matrix_fragment_80),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                      select_half_operand<2, false, false>(matrix_fragment_82),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_44 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_fragment_81),
                                      select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_accumulator_45 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                                      select_half_operand<2, false, false>(matrix_fragment_83),
                                      select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(packed_half_114));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_fragment_76, matrix_fragment_77, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                       select_half_operand<2, false, false>(matrix_accumulator_44));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_46), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_88, matrix_accumulator_89, matrix_fragment_42,
                             matrix_fragment_43);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_47), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    uniform_block_index_16 = blockIdx.x;

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_56,
                             matrix_fragment_57, matrix_fragment_58, matrix_fragment_59,
                             matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_48,
                             matrix_fragment_49);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_48 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_12) >> 0x1fu);

    uniform_address_word_6 = uint32_t(int32_t(uniform_parameter_word_13) >> 0x1fu);

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(packed_half_115));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(packed_half_115));

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_12;

    uniform_index_word_8 = uint32_t(combine_words(uniform_address_word_6, 0u) << 0x2u >> 32) +
                           uniform_parameter_word_13;

    matrix_fragment_72 = add_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(packed_half_114));

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_8) >> 0x2u);

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_88, matrix_accumulator_89, matrix_accumulator_44,
                             matrix_accumulator_45);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_72), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_75 = result_word;
    }

    uniform_index_word_12 = (uniform_index_word_11 << 0x3u) + uniform_parameter_word_15;

    uniform_parameter_word_14 = (uniform_block_index_16 << 0x3u) + uniform_parameter_word_14;

    uniform_condition_1 = int32_t(uniform_index_word_9) >= int32_t(uniform_index_word_5);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_73), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_74 = result_word;
    }

    uniform_condition_0 = int32_t(uniform_index_word_10) >= int32_t(uniform_index_word_8);

    uniform_address_word_6 = parameters[parameter_output_low_2];
    uniform_address_word_7 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_77), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_88 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_76), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_89 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_60,
                             matrix_fragment_61, matrix_fragment_62, matrix_fragment_63,
                             matrix_accumulator_90, matrix_accumulator_91, matrix_fragment_52,
                             matrix_fragment_53);

    uniform_condition_1 =
        int32_t(uniform_index_word_12) > int32_t(0u - 0x4u) && !uniform_condition_1;

    uniform_condition_0 =
        int32_t(uniform_parameter_word_14) > int32_t(0u - 0x4u) && !uniform_condition_0;

    uniform_condition_2 = uniform_condition_1 & uniform_condition_0;

    uniform_index_word_11 = uniform_index_word_9 * uniform_index_word_8 + uniform_index_word_10;

    uniform_index_word_11 = uniform_index_word_11 << (0x7u & 31u);

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_11)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));
    if (uniform_condition_2) {
        matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                           select_half_operand<2, false, false>(0u));

        split_address_words(packed_half_46, matrix_fragment_47,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_57),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_56),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        split_address_words(packed_half_46, matrix_fragment_47,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_46, matrix_fragment_47));

        matrix_fragment_43 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                          select_half_operand<2, false, false>(matrix_fragment_48));

        matrix_fragment_40 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                          select_half_operand<2, false, false>(matrix_fragment_49));

        matrix_fragment_42 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                          select_half_operand<2, false, false>(matrix_fragment_51));

        store_four_words_streaming<CheckBounds>(combine_words(packed_half_46, matrix_fragment_47),
                                                matrix_fragment_40, matrix_fragment_41,
                                                matrix_fragment_42, matrix_fragment_43,
                                                memory_ranges, error_counts);
    }
    uniform_index_word_11 = uniform_index_word_10 + 0x1u;

    split_address_words(matrix_fragment_40, matrix_fragment_41,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_2 = int32_t(uniform_index_word_11) >= int32_t(uniform_index_word_8);

    matrix_fragment_64 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_64),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_65),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_2 =
        int32_t(uniform_parameter_word_14) > int32_t(0u - 0x8u) && !uniform_condition_2;

    packed_half_46 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                   select_half_operand<2, false, false>(0u));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_60),
                                       select_half_operand<2, false, false>(0u));

    uniform_condition_1 = uniform_condition_1 & uniform_condition_2;

    matrix_accumulator_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                       select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                       select_half_operand<2, false, false>(matrix_accumulator_74));

    uniform_address_word_6 = parameters[parameter_output_low_2];
    uniform_address_word_7 = parameters[parameter_output_high_3];

    condition_4 = uniform_condition_1;

    if (condition_4) {
        split_address_words(matrix_fragment_78, matrix_fragment_79,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_40, matrix_fragment_41));

        matrix_fragment_43 = permute_signed(matrix_fragment_43, packed_half_46, 0x5410u);

        matrix_fragment_42 = permute_signed(matrix_fragment_42, matrix_fragment_47, 0x5410u);

        matrix_fragment_41 = permute_signed(matrix_fragment_65, matrix_accumulator_44, 0x5410u);

        matrix_fragment_40 = permute_signed(matrix_fragment_64, matrix_accumulator_45, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_78, matrix_fragment_79) + 0x200ull, matrix_fragment_40,
            matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges,
            error_counts);
    }

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x46a0ull, matrix_accumulator_44,
        matrix_accumulator_45, packed_half_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x48a0ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x4aa0ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x4ca0ull, matrix_fragment_56,
        matrix_fragment_57, matrix_fragment_58, matrix_fragment_59, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x44a0ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x4ea0ull, matrix_fragment_60,
        matrix_fragment_61, matrix_fragment_62, matrix_fragment_63, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x50a0ull, matrix_fragment_64,
        matrix_fragment_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x52a0ull, matrix_accumulator_68,
        matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    matrix_accumulator_75 = permute_signed(matrix_fragment_72, matrix_fragment_72, 0x5432u);

    matrix_fragment_78 = permute_signed(matrix_fragment_73, matrix_fragment_73, 0x5432u);

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72),
                       select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_73 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                                        select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_fragment_72 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_72), 0x4100410u);

    matrix_fragment_73 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73), 0x4100410u);

    matrix_accumulator_74 = half_to_float_bits((matrix_fragment_72 & 65535u) * 65537u);

    matrix_fragment_78 = half_to_float_bits((matrix_fragment_73 & 65535u) * 65537u);

    matrix_fragment_73 = half_to_float_bits((matrix_fragment_73 >> 16) * 65537u);

    matrix_fragment_72 = half_to_float_bits((matrix_fragment_72 >> 16) * 65537u);

    matrix_accumulator_74 = rsqrt_bits(matrix_accumulator_74);

    matrix_accumulator_75 = rsqrt_bits(matrix_fragment_72);

    matrix_fragment_78 = rsqrt_bits(matrix_fragment_78);

    matrix_fragment_73 = rsqrt_bits(matrix_fragment_73);

    matrix_accumulator_75 = pack_half(matrix_accumulator_75, matrix_accumulator_74);

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_79 = pack_half(matrix_fragment_73, matrix_fragment_78);

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<2, false, false>(matrix_fragment_79));

    matrix_accumulator_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_96),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_86 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_86),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_97),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_87 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_87),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_94 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_92 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_accumulator_93 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_96),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_86),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_97 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_97),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_74 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_87),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_94 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_75 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_78 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = permute_signed(matrix_fragment_72, matrix_fragment_73, 0x5410u);

    matrix_fragment_73 = permute_signed(matrix_accumulator_97, matrix_accumulator_74, 0x5410u);

    matrix_accumulator_74 = permute_signed(matrix_accumulator_94, matrix_accumulator_75, 0x5410u);

    matrix_accumulator_75 = permute_signed(matrix_accumulator_95, matrix_fragment_78, 0x5410u);

    matrix_fragment_77 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                       select_half_operand<2, false, false>(matrix_accumulator_88));

    matrix_fragment_76 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                       select_half_operand<2, false, false>(matrix_accumulator_89));

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_40,
                             matrix_fragment_41);

    matrix_fragment_40 = permute_signed(matrix_fragment_77, matrix_fragment_77, 0x5432u);

    matrix_fragment_41 = permute_signed(matrix_fragment_76, matrix_fragment_76, 0x5432u);

    matrix_fragment_77 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_76 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x4100410u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x4100410u);

    matrix_fragment_40 = half_to_float_bits((matrix_fragment_77 & 65535u) * 65537u);

    matrix_fragment_77 = half_to_float_bits((matrix_fragment_77 >> 16) * 65537u);

    matrix_fragment_41 = half_to_float_bits((matrix_fragment_76 & 65535u) * 65537u);

    matrix_fragment_76 = half_to_float_bits((matrix_fragment_76 >> 16) * 65537u);

    matrix_fragment_40 = rsqrt_bits(matrix_fragment_40);

    matrix_fragment_77 = rsqrt_bits(matrix_fragment_77);

    matrix_fragment_41 = rsqrt_bits(matrix_fragment_41);

    matrix_fragment_76 = rsqrt_bits(matrix_fragment_76);

    multiply_accumulate_half(matrix_accumulator_86, matrix_accumulator_87, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_42,
                             matrix_fragment_43);

    matrix_fragment_40 = pack_half(matrix_fragment_77, matrix_fragment_40);

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                          select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_42 = pack_half(matrix_fragment_76, matrix_fragment_41);

    packed_half_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                          select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<2, false, false>(matrix_fragment_42));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                          select_half_operand<2, false, false>(matrix_fragment_42));

    packed_half_121 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                                          select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_83 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                            select_half_operand<0, false, false>(packed_half_2));

    packed_half_116 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_116),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_120 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_120),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_117 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_117),
                                          select_half_operand<0, false, false>(packed_half_2));

    packed_half_121 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_121),
                                          select_half_operand<0, false, false>(packed_half_2));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_44, matrix_accumulator_45, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_44,
                             matrix_accumulator_45);

    matrix_fragment_81 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                       select_half_operand<2, false, false>(0u));

    packed_half_116 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_116),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_120),
                                       select_half_operand<2, false, false>(0u));

    packed_half_117 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_117),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_40 = permute_signed(matrix_fragment_40, matrix_fragment_41, 0x5410u);

    matrix_fragment_41 = permute_signed(matrix_fragment_81, matrix_fragment_42, 0x5410u);

    multiply_accumulate_half(packed_half_46, matrix_fragment_47, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_16, matrix_fragment_17, packed_half_46,
                             matrix_fragment_47);

    matrix_fragment_42 = permute_signed(packed_half_116, matrix_fragment_43, 0x5410u);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_48,
                             matrix_fragment_49);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_50,
                             matrix_fragment_51);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_72,
                             matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75,
                             matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_121),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = permute_signed(packed_half_117, matrix_fragment_72, 0x5410u);

    multiply_accumulate_half(matrix_fragment_56, matrix_fragment_57, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_22, matrix_fragment_23, matrix_fragment_56,
                             matrix_fragment_57);

    multiply_accumulate_half(matrix_fragment_58, matrix_fragment_59, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_20, matrix_fragment_21, matrix_fragment_58,
                             matrix_fragment_59);

    multiply_accumulate_half(matrix_fragment_60, matrix_fragment_61, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_60,
                             matrix_fragment_61);

    multiply_accumulate_half(matrix_fragment_62, matrix_fragment_63, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_62,
                             matrix_fragment_63);

    matrix_accumulator_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_86 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_46),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_45 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x3c203c20u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_accumulator_87 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_86 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3c203c20u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_44 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x3c203c20u);

    matrix_fragment_56 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_56),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x3c203c20u);

    matrix_fragment_58 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_58),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_60),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_45 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_45), 0x3e473e47u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_62),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_86 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_86), 0x3e473e47u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_44 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_44), 0x3e473e47u);

    matrix_fragment_50 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_50),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_46), 0x3e473e47u);

    matrix_fragment_51 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_51),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_accumulator_87 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3c203c20u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_56 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3c203c20u);

    matrix_fragment_58 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3c203c20u);

    matrix_fragment_60 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3c203c20u);

    matrix_fragment_62 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3c203c20u);

    matrix_fragment_23 = (matrix_accumulator_45 << 0x5u) + 0x7ff88000u;

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_accumulator_87 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_87), 0x3e473e47u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56), 0x3e473e47u);

    matrix_fragment_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58), 0x3e473e47u);

    matrix_fragment_60 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60), 0x3e473e47u);

    matrix_fragment_62 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62), 0x3e473e47u);

    matrix_fragment_18 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_fragment_21 = (matrix_accumulator_86 << 0x5u) + 0x7ff88000u;

    matrix_fragment_22 = (matrix_accumulator_44 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_45 = (packed_half_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3c203c20u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3c203c20u);

    matrix_fragment_50 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3c203c20u);

    matrix_fragment_51 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_fragment_19 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_fragment_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_63),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_20 = (matrix_accumulator_87 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_44 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = (matrix_fragment_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                        select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_17 = (matrix_fragment_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                       select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3e473e47u);

    matrix_fragment_49 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3e473e47u);

    matrix_fragment_50 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_68,
                             matrix_accumulator_69);

    matrix_fragment_51 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51), 0x3e473e47u);

    matrix_fragment_12 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_63 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3c203c20u);

    matrix_fragment_10 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 = (matrix_fragment_58 << 0x5u) + 0x7ff88000u;

    packed_half_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                    select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_fragment_64, matrix_fragment_65, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_64,
                             matrix_fragment_65);

    matrix_fragment_63 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63), 0x3e473e47u);

    matrix_fragment_56 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                        select_half_operand<2, false, false>(packed_half_46));

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                        select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_15 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                       select_half_operand<2, false, false>(matrix_accumulator_44));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_84, matrix_accumulator_85, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_14 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_fragment_40 = (matrix_fragment_48 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_fragment_49 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = (matrix_fragment_50 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_fragment_51 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_15 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_51 = (matrix_fragment_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                        select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                        select_half_operand<2, false, false>(matrix_fragment_15));

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x54b0ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_160, address_word_161) + 0x56b0ull, address_word_160,
        address_word_161, matrix_fragment_162, matrix_fragment_163, memory_ranges, error_counts);

    matrix_fragment_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_64),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_57 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_57),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3c203c20u);

    matrix_fragment_59 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_59),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_61),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_57 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3c203c20u);

    matrix_fragment_64 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_64), 0x3e473e47u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_65),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_59 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3c203c20u);

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_61 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3c203c20u);

    matrix_fragment_57 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57), 0x3e473e47u);

    matrix_fragment_53 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_53),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    packed_half_46 = (matrix_fragment_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_54 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_54),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_47 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_55 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_55),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_59 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59), 0x3e473e47u);

    matrix_fragment_52 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_52),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_61 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61), 0x3e473e47u);

    matrix_accumulator_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_65 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3c203c20u);

    matrix_accumulator_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_accumulator_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_accumulator_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71),
        select_half_operand<1, false, false>(packed_half_2), 0x3d343d34u);

    matrix_fragment_48 = (matrix_fragment_57 << 0x5u) + 0x7ff88000u;

    matrix_fragment_57 = add_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_49 = (matrix_fragment_59 << 0x5u) + 0x7ff88000u;

    matrix_fragment_50 = (matrix_fragment_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3c203c20u);

    matrix_fragment_54 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3c203c20u);

    matrix_fragment_55 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3c203c20u);

    matrix_fragment_65 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_65), 0x3e473e47u);

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3c203c20u);

    matrix_accumulator_68 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3c203c20u);

    matrix_accumulator_69 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3c203c20u);

    matrix_accumulator_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3c203c20u);

    matrix_accumulator_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3c203c20u);

    matrix_fragment_66 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_56 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53), 0x3e473e47u);

    matrix_fragment_58 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                        select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_54 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54), 0x3e473e47u);

    matrix_fragment_59 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                        select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_55 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55), 0x3e473e47u);

    matrix_fragment_52 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x3e473e47u);

    matrix_accumulator_68 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3e473e47u);

    matrix_accumulator_69 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3e473e47u);

    matrix_accumulator_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3e473e47u);

    matrix_accumulator_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3e473e47u);

    packed_half_2 = (matrix_fragment_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_53 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_57 = (matrix_fragment_54 << 0x5u) + 0x7ff88000u;

    matrix_fragment_61 = (matrix_fragment_55 << 0x5u) + 0x7ff88000u;

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_52 = (matrix_fragment_52 << 0x5u) + 0x7ff88000u;

    matrix_fragment_56 = (matrix_fragment_56 << 0x5u) + 0x7ff88000u;

    matrix_fragment_54 = (matrix_accumulator_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_55 = (matrix_accumulator_69 << 0x5u) + 0x7ff88000u;

    matrix_fragment_59 = (matrix_accumulator_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_58 = (matrix_accumulator_71 << 0x5u) + 0x7ff88000u;

    matrix_fragment_67 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_62),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_62 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_57));

    matrix_fragment_64 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                                        select_half_operand<2, false, false>(matrix_fragment_61));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                        select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                       select_half_operand<2, false, false>(matrix_fragment_58));

    matrix_fragment_60 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_60),
                                        select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_63 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_63),
                                        select_half_operand<2, false, false>(matrix_fragment_64));

    matrix_fragment_65 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66),
                                        select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                       select_half_operand<2, false, false>(matrix_accumulator_68));

    matrix_fragment_62 = condition_0 ? matrix_fragment_63 : matrix_fragment_60;

    matrix_fragment_60 = condition_0 ? matrix_fragment_60 : matrix_fragment_63;

    matrix_fragment_67 = condition_0 ? matrix_accumulator_68 : matrix_fragment_65;

    matrix_fragment_65 = condition_0 ? matrix_fragment_65 : matrix_accumulator_68;

    matrix_fragment_63 = !condition_2 ? matrix_fragment_62 : matrix_fragment_67;

    matrix_fragment_64 = !condition_2 ? matrix_fragment_60 : matrix_fragment_65;

    matrix_fragment_67 = !condition_2 ? matrix_fragment_67 : matrix_fragment_62;

    matrix_fragment_60 = !condition_2 ? matrix_fragment_65 : matrix_fragment_60;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_63), "r"(matrix_fragment_4), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_63 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_64), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_64 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_67), "r"(matrix_fragment_6), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_60), "r"(matrix_fragment_7), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_60 = result_word;
    }

    matrix_fragment_62 = !condition_3 ? matrix_fragment_63 : matrix_fragment_64;

    matrix_fragment_65 = !condition_3 ? matrix_fragment_64 : matrix_fragment_63;

    matrix_accumulator_69 = !condition_3 ? matrix_fragment_67 : matrix_fragment_60;

    matrix_fragment_66 = !condition_3 ? matrix_fragment_60 : matrix_fragment_67;

    matrix_accumulator_68 = !condition_1 ? matrix_fragment_62 : matrix_accumulator_69;

    matrix_accumulator_71 = !condition_1 ? matrix_fragment_65 : matrix_fragment_66;

    matrix_fragment_62 = !condition_1 ? matrix_accumulator_69 : matrix_fragment_62;

    matrix_fragment_65 = !condition_1 ? matrix_fragment_66 : matrix_fragment_65;

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                       select_half_operand<2, false, false>(matrix_accumulator_71));

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                       select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                       select_half_operand<2, false, false>(matrix_fragment_65));

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_68),
                       select_half_operand<1, false, false>(matrix_accumulator_68));

    matrix_accumulator_68 = permute_signed(matrix_accumulator_68, matrix_accumulator_68, 0x5410u);

    matrix_accumulator_68 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68), 0x4100410u);

    matrix_fragment_4 = half_to_float_bits((matrix_accumulator_68 & 65535u) * 65537u);

    matrix_accumulator_68 = half_to_float_bits((matrix_accumulator_68 >> 16) * 65537u);

    matrix_fragment_4 = reciprocal_bits(matrix_fragment_4);

    matrix_fragment_7 = uint32_t(int32_t(matrix_fragment_8) >> 0x1fu);

    matrix_fragment_5 = reciprocal_bits(matrix_accumulator_68);

    matrix_fragment_7 =
        uint32_t(combine_words(matrix_fragment_7, 0u) << 0x5u >> 32) + matrix_fragment_8;

    matrix_fragment_7 = matrix_fragment_7 & 0xffffffe0u;

    matrix_fragment_6 = pack_half(matrix_fragment_5, matrix_fragment_4);

    matrix_fragment_5 = matrix_fragment_8 + (0u - matrix_fragment_7);

    matrix_fragment_7 = uint32_t(int64_t(int32_t(packed_half_3)) * (1ll << 0x1eu) >> 32) + 0x10u;

    packed_half_3 = uint32_t(int64_t(int32_t(packed_half_3)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_4 = uint32_t(int32_t(matrix_fragment_7) >> 0x1fu);

    matrix_fragment_8 = uint32_t(int32_t(packed_half_3) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_9), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_9 = result_word;
    }

    matrix_fragment_4 =
        uint32_t(combine_words(matrix_fragment_4, 0u) << 0x5u >> 32) + matrix_fragment_7;

    matrix_fragment_8 =
        uint32_t(combine_words(matrix_fragment_8, 0u) << 0x5u >> 32) + packed_half_3;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_5), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_5 = result_word;
    }

    matrix_fragment_4 = matrix_fragment_4 & 0xffffffe0u;

    matrix_fragment_8 = matrix_fragment_8 & 0xffffffe0u;

    matrix_fragment_65 = matrix_fragment_7 + (0u - matrix_fragment_4);

    matrix_fragment_63 = packed_half_3 + (0u - matrix_fragment_8);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_65), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_62 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_6), "r"(matrix_fragment_63), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_63 = result_word;
    }

    packed_half_3 = permute_signed(matrix_fragment_9, matrix_fragment_9, 0x5410u);

    matrix_fragment_60 = permute_signed(matrix_fragment_5, matrix_fragment_5, 0x5410u);

    matrix_fragment_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_21),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_45),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_20),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_44),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_21 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_22 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_22),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_accumulator_45 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_45),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_accumulator_44 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_44),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_62 = permute_signed(matrix_fragment_62, matrix_fragment_62, 0x5410u);

    matrix_fragment_63 = permute_signed(matrix_fragment_63, matrix_fragment_63, 0x5410u);

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_21));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_20));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(matrix_accumulator_45));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_44));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_100, packed_half_101, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_102, packed_half_103, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_6),
                                      select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(packed_half_3));

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, packed_half_100, packed_half_101,
                             0u, 0u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_52 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                            select_half_operand<2, false, false>(packed_half_3));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_57),
                                        select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_61),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_104, matrix_fragment_105, 0u, 0u);

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_59),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_53 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_61),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_accumulator_98, matrix_accumulator_99, 0u, 0u);

    matrix_fragment_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_58),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    packed_half_46 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_46),
                                         select_half_operand<2, false, false>(matrix_fragment_62));

    multiply_accumulate_half(matrix_fragment_102, packed_half_103, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_102, packed_half_103, 0u, 0u);

    matrix_fragment_54 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                            select_half_operand<2, false, false>(matrix_fragment_62));

    matrix_fragment_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_56),
                            select_half_operand<2, false, false>(matrix_fragment_60));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_43));

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_2),
                                        select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                      select_half_operand<2, false, false>(matrix_fragment_42));

    matrix_fragment_55 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                            select_half_operand<2, false, false>(matrix_fragment_63));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_7),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_59 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_59),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_58 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_58),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_112, packed_half_113, matrix_fragment_22,
                             matrix_fragment_23);

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_46),
                                       select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                       select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_59));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_58));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_108, packed_half_109, matrix_fragment_18,
                             matrix_fragment_19);

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             packed_half_110, packed_half_111, matrix_fragment_8,
                             matrix_fragment_9);

    multiply_accumulate_half(matrix_fragment_20, matrix_fragment_21, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_106, matrix_fragment_107, matrix_fragment_20,
                             matrix_fragment_21);

    multiply_accumulate_half(packed_half_100, packed_half_101, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_108, packed_half_109, packed_half_100, packed_half_101);

    multiply_accumulate_half(matrix_fragment_104, matrix_fragment_105, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_112, packed_half_113, matrix_fragment_104,
                             matrix_fragment_105);

    multiply_accumulate_half(matrix_accumulator_98, matrix_accumulator_99, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_106, matrix_fragment_107, matrix_accumulator_98,
                             matrix_accumulator_99);

    multiply_accumulate_half(matrix_fragment_102, packed_half_103, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             packed_half_110, packed_half_111, matrix_fragment_102,
                             packed_half_103);

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                      select_half_operand<2, false, false>(0u));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_7));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_20),
                                      select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_21),
                                      select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    packed_half_3 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                  select_half_operand<2, false, false>(0u));

    packed_half_2 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_104),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_105),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_119));

    packed_half_100 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_98),
                                    select_half_operand<2, false, false>(packed_half_3));

    matrix_fragment_10 = multiply_half_pairs(select_half_operand<2, false, false>(packed_half_38),
                                             select_half_operand<2, false, false>(packed_half_119));

    packed_half_101 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_99),
                                    select_half_operand<2, false, false>(packed_half_2));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(packed_half_119));

    matrix_fragment_102 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_102),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(packed_half_119));

    packed_half_103 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_103),
                                    select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_16 = threadIdx.x;

    matrix_fragment_17 = threadIdx.y;

    matrix_fragment_18 = threadIdx.z;

    multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_10,
                             matrix_fragment_11);

    packed_half_2 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(packed_half_118));

    packed_half_3 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_26),
                            select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_27),
                            select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(packed_half_115));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, packed_half_100,
                             packed_half_101, matrix_fragment_102, packed_half_103,
                             matrix_fragment_12, matrix_fragment_13, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(packed_half_115));

    matrix_fragment_24 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_24),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_25 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_25),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_28 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(packed_half_114));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(packed_half_114));

    uniform_index_word_11 = uniform_index_word_9 + 0x1u;

    uniform_condition_1 = int32_t(uniform_index_word_11) >= int32_t(uniform_index_word_5);

    multiply_accumulate_half(packed_half_2, packed_half_3, matrix_fragment_4, matrix_fragment_5,
                             matrix_fragment_6, matrix_fragment_7, matrix_fragment_14,
                             matrix_fragment_15, packed_half_2, packed_half_3);

    uniform_condition_1 =
        int32_t(uniform_index_word_12) > int32_t(0u - 0x8u) && !uniform_condition_1;

    uniform_index_word_8 = uniform_index_word_9 * uniform_index_word_8 + uniform_index_word_8;

    uniform_condition_0 = uniform_condition_1 & uniform_condition_0;

    multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, packed_half_100, packed_half_101,
                             matrix_fragment_102, packed_half_103, address_word_160,
                             address_word_161, matrix_fragment_8, matrix_fragment_9);

    uniform_index_word_8 = uniform_index_word_10 + uniform_index_word_8;

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, packed_half_100,
                             packed_half_101, matrix_fragment_102, packed_half_103,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(address_word_160, address_word_161, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             address_word_160, address_word_161, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(packed_half_100, packed_half_101, packed_half_100, packed_half_101,
                             matrix_fragment_102, packed_half_103, matrix_fragment_162,
                             matrix_fragment_163, matrix_fragment_24, matrix_fragment_25);

    multiply_accumulate_half(matrix_fragment_4, matrix_fragment_5, matrix_fragment_4,
                             matrix_fragment_5, matrix_fragment_6, matrix_fragment_7,
                             matrix_fragment_162, matrix_fragment_163, matrix_fragment_28,
                             matrix_fragment_29);

    condition_0 = (matrix_fragment_16 | matrix_fragment_17 | matrix_fragment_18) != 0;

    uniform_index_word_8 = uniform_index_word_8 << (0x7u & 31u);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_8)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));
    if (uniform_condition_0) {
        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_2),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_3),
                                          select_half_operand<2, false, false>(0u));

        split_address_words(packed_half_2, packed_half_3,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_5),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_4),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                          select_half_operand<2, false, false>(matrix_fragment_6));

        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_161),
                                          select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_160),
                                          select_half_operand<2, false, false>(matrix_fragment_19));

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, packed_half_3), matrix_fragment_4, matrix_fragment_5,
            matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    uniform_condition_1 = uniform_condition_1 & uniform_condition_2;

    matrix_fragment_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_100),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_101),
                                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_1) {
        split_address_words(packed_half_2, packed_half_3,
                            combine_words(uniform_address_word_6, uniform_address_word_7));

        matrix_fragment_5 = permute_signed(matrix_fragment_5, matrix_fragment_4, 0x5410u);

        matrix_fragment_7 = permute_signed(matrix_fragment_7, matrix_fragment_8, 0x5410u);

        matrix_fragment_6 = permute_signed(matrix_fragment_6, matrix_fragment_9, 0x5410u);

        split_address_words(packed_half_2, packed_half_3,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_2, packed_half_3));

        matrix_fragment_4 = permute_signed(matrix_fragment_16, matrix_fragment_11, 0x5410u);

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_2, packed_half_3) + 0x200ull, matrix_fragment_4,
            matrix_fragment_5, matrix_fragment_6, matrix_fragment_7, memory_ranges, error_counts);
    }
    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_address_word_6 = parameters[parameter_publication_optional_low_14];
    uniform_address_word_7 = parameters[parameter_publication_optional_high_15];

    uniform_index_word_5 = uniform_block_index_16 + uniform_address_word_4;

    split_address_words(uniform_address_word_6, uniform_address_word_7,
                        int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_address_word_6, uniform_address_word_7));

    split_address_words(packed_half_2, packed_half_3,
                        combine_words(uniform_address_word_6, uniform_address_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(packed_half_2, packed_half_3), 0u, memory_ranges,
                                error_counts);
    return;
}
extern "C" __global__ __launch_bounds__(32, 14) void phoenix_window_attention_channels_32_upsample(
    KernelArguments parameters) {
    run_window_attention_channels_32_upsample<false>(parameters.words, nullptr, nullptr);
}
