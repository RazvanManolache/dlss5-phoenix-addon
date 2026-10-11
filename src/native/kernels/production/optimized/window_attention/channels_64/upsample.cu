// DLSS5 Phoenix optimized kernel: window_attention/channels_64/upsample.cu
// Fused shifted-window block: feed-forward projections, 32-channel attention heads and residual publication.
// The channel folder fixes the specialization; its down/up/view variant controls spatial layout.
// Packed word variables hold register bits, sometimes with several uses over their lifetime.
// FP16/FP8 rounding, atomic update order and publication fences are part of the numerical contract.
// Predicate masks use eager Boolean operators (& and |) to avoid short-circuit control flow.

#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../../common/texture_sampling.cuh"
#define DLSS5_ENABLE_ASYNC_COPIES 1
#define DLSS5_TILE_SHARED_BYTES 5120
#include "../../../common/device_memory_scalar.cuh"
// Low/high parameter pairs form 64-bit addresses; integer slots carry geometry.
enum ParameterWord : unsigned {
    parameter_input_low_0 = 0,
    parameter_input_high_1 = 1,
    parameter_output_low_2 = 2,
    parameter_output_high_3 = 3,
    parameter_trained_weights_low_4 = 4,
    parameter_trained_weights_high_5 = 5,
    parameter_input_upsample_skip_optional_low_6 = 6,
    parameter_input_upsample_skip_optional_high_7 = 7,
    parameter_tensor_height_8 = 8,
    parameter_tensor_width_9 = 9,
    parameter_optional_control_10 = 10,
    parameter_optional_control_11 = 11,
    parameter_publication_optional_low_16 = 16,
    parameter_publication_optional_high_17 = 17,
};

struct alignas(8) KernelArguments {
    uint32_t words[22];
};
static_assert(sizeof(KernelArguments) == 88, "Native ABI");
template <bool CheckBounds>
__device__ __forceinline__ void run_window_attention_channels_64_upsample(
    const uint32_t* parameters, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    uint32_t address_word_0 = 0;
    uint32_t reused_word_1 = 0;
    uint32_t matrix_accumulator_2 = 0;
    uint32_t matrix_accumulator_3 = 0;
    uint32_t packed_half_4 = 0;
    uint32_t packed_half_5 = 0;
    uint32_t packed_half_6 = 0;
    uint32_t matrix_accumulator_7 = 0;
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
    uint32_t matrix_accumulator_20 = 0;
    uint32_t matrix_accumulator_21 = 0;
    uint32_t matrix_fragment_22 = 0;
    uint32_t matrix_fragment_23 = 0;
    uint32_t address_word_24 = 0;
    uint32_t address_word_25 = 0;
    uint32_t matrix_accumulator_26 = 0;
    uint32_t matrix_accumulator_27 = 0;
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
    uint32_t matrix_accumulator_56 = 0;
    uint32_t matrix_accumulator_57 = 0;
    uint32_t matrix_accumulator_58 = 0;
    uint32_t matrix_accumulator_59 = 0;
    uint32_t matrix_accumulator_60 = 0;
    uint32_t matrix_accumulator_61 = 0;
    uint32_t matrix_accumulator_62 = 0;
    uint32_t matrix_accumulator_63 = 0;
    uint32_t matrix_accumulator_64 = 0;
    uint32_t matrix_accumulator_65 = 0;
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
    uint32_t packed_half_118 = 0;
    uint32_t matrix_accumulator_119 = 0;
    uint32_t matrix_fragment_120 = 0;
    uint32_t matrix_fragment_121 = 0;
    uint32_t matrix_fragment_122 = 0;
    uint32_t matrix_fragment_123 = 0;
    uint32_t matrix_accumulator_124 = 0;
    uint32_t matrix_accumulator_125 = 0;
    uint32_t matrix_fragment_126 = 0;
    uint32_t matrix_fragment_127 = 0;
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
    uint32_t packed_half_140 = 0;
    uint32_t matrix_accumulator_141 = 0;
    uint32_t matrix_accumulator_142 = 0;
    uint32_t matrix_accumulator_143 = 0;
    uint32_t matrix_fragment_144 = 0;
    uint32_t matrix_fragment_145 = 0;
    uint32_t matrix_fragment_146 = 0;
    uint32_t matrix_fragment_147 = 0;
    uint32_t matrix_accumulator_148 = 0;
    uint32_t matrix_accumulator_149 = 0;
    uint32_t matrix_accumulator_150 = 0;
    uint32_t matrix_accumulator_151 = 0;
    uint32_t packed_half_152 = 0;
    uint32_t address_word_153 = 0;
    uint32_t packed_half_154 = 0;
    uint32_t packed_half_155 = 0;
    uint32_t matrix_accumulator_156 = 0;
    uint32_t matrix_accumulator_157 = 0;
    uint32_t packed_half_158 = 0;
    uint32_t index_word_159 = 0;
    uint32_t packed_half_160 = 0;
    uint32_t address_word_161 = 0;
    uint32_t packed_half_162 = 0;
    uint32_t address_word_163 = 0;
    uint32_t reused_word_164 = 0;
    uint32_t reused_word_165 = 0;
    uint32_t uniform_address_word_4 = 0;
    uint32_t uniform_index_word_5 = 0;
    uint32_t uniform_index_word_6 = 0;
    uint32_t uniform_index_word_7 = 0;
    uint32_t uniform_index_word_8 = 0;
    uint32_t uniform_index_word_9 = 0;
    uint32_t uniform_index_word_10 = 0;
    uint32_t uniform_index_word_11 = 0;
    uint32_t uniform_index_word_12 = 0;
    uint32_t uniform_index_word_13 = 0;
    uint32_t uniform_index_word_14 = 0;
    uint32_t uniform_index_word_15 = 0;
    uint32_t uniform_reused_word_16 = 0;
    uint32_t uniform_reused_word_17 = 0;
    uint32_t uniform_index_word_18 = 0;
    uint32_t uniform_index_word_19 = 0;
    uint32_t uniform_parameter_word_20 = 0;
    uint32_t uniform_parameter_word_21 = 0;
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

    uniform_address_word_4 = parameters[parameter_tensor_height_8];

    address_word_0 = threadIdx.x & 31u;

    uniform_parameter_word_21 = blockIdx.y;

    packed_half_6 = 0;
    matrix_accumulator_7 = 0;

    uniform_index_word_15 = parameters[parameter_tensor_width_9];

    matrix_accumulator_20 = threadIdx.y;

    matrix_fragment_8 = 0;
    matrix_fragment_9 = 0;

    matrix_fragment_10 = 0;
    matrix_fragment_11 = 0;

    uniform_index_word_10 = parameters[parameter_optional_control_10];
    uniform_index_word_11 = parameters[parameter_optional_control_11];

    matrix_fragment_40 = threadIdx.x & 31u;

    uniform_index_word_7 = blockIdx.x;

    uniform_reused_word_16 = 0u;
    uniform_reused_word_17 = 0u;

    uniform_address_word_4 = uniform_address_word_4 + 0x1u;

    matrix_fragment_12 = parameters[parameter_trained_weights_low_4];
    matrix_fragment_13 = parameters[parameter_trained_weights_high_5];

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_address_word_4, 0u) << 0x1u >> 32) + uniform_address_word_4;

    uniform_index_word_8 =
        uint32_t(int64_t(int32_t(uniform_address_word_4)) * (1ll << 0x1fu) >> 32) + 0x3u;

    uniform_index_word_6 = (uniform_parameter_word_21 << 0x3u) + uniform_index_word_11;

    uniform_index_word_5 = min(int32_t(0u), int32_t(uniform_index_word_10));

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_6, 0u) << 0x1u >> 32) + uniform_index_word_6;

    matrix_accumulator_2 = address_word_0;

    uniform_index_word_14 =
        uint32_t(combine_words(uniform_index_word_9, 0u) << 0x2u >> 32) + uniform_index_word_8;

    uniform_index_word_8 = uniform_index_word_15 + 0x1u;

    uniform_index_word_7 = (uniform_index_word_7 << 0x3u) + uniform_index_word_10;

    uniform_index_word_9 = 0u - uniform_index_word_5 + uniform_index_word_15;

    matrix_accumulator_3 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    uniform_index_word_8 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x1u >> 32) + uniform_index_word_8;

    uniform_index_word_12 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    uniform_address_word_4 =
        uint32_t(combine_words(uniform_index_word_7, 0u) << 0x1u >> 32) + uniform_index_word_7;

    uniform_condition_0 = uint32_t(uniform_index_word_9) != 0u;

    uniform_index_word_8 =
        uint32_t(int64_t(int32_t(uniform_index_word_8)) * (1ll << 0x1fu) >> 32) + 0x3u;

    packed_half_4 =
        uint32_t(combine_words(matrix_accumulator_3, 0u) << 0x2u >> 32) + matrix_accumulator_3;

    uniform_index_word_13 = uint32_t(int32_t(uniform_address_word_4) >> 0x1u);

    matrix_fragment_28 = address_word_0;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_8) >> 0x1fu);

    packed_half_4 = packed_half_4 & 0xfffffffcu;

    uniform_index_word_18 = uniform_index_word_14 & 0xfffffffcu;

    packed_half_5 = uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                    uniform_index_word_12;

    uniform_index_word_15 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_8;

    matrix_accumulator_21 = matrix_accumulator_3 + uniform_index_word_13 + (0u - packed_half_4);

    if (uniform_condition_0) {
        uniform_index_word_10 = uniform_index_word_9 + 0xffffffffu;
    }

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_6) >> 0x1fu);

    uniform_index_word_19 = uniform_index_word_15 & 0xfffffffcu;

    condition_0 = int32_t(packed_half_5) >= int32_t(uniform_index_word_18);

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_10) >> 0x1fu);
    }

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_accumulator_20)) * uint64_t(uint32_t(0x400u)) +
                            combine_words(matrix_fragment_12, matrix_fragment_13));

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_7) >> 0x1fu);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_6;

    condition_1 = int32_t(matrix_accumulator_21) >= int32_t(uniform_index_word_19);

    condition_0 = int32_t(packed_half_5) > -1 && !condition_0;

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(combine_words(uniform_index_word_11, 0u) << 0x3u >> 32) +
                                uniform_index_word_10;
    }

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x7200u);

    condition_1 = int32_t(matrix_accumulator_21) > -1 && !condition_1;

    uniform_index_word_9 =
        uint32_t(combine_words(uniform_index_word_8, 0u) << 0x2u >> 32) + uniform_index_word_7;

    uniform_index_word_15 = uniform_index_word_15 << (0x2u & 31u);

    uniform_index_word_14 = uniform_index_word_14 << (0x1u & 31u);

    if (uniform_condition_0) {
        uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x3u);
    }

    packed_half_4 = 0;
    packed_half_5 = 0;

    condition_0 = condition_0 & condition_1;

    uniform_address_word_4 = 0u;

    matrix_accumulator_3 = uniform_index_word_12;

    uniform_index_word_8 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    uniform_index_word_9 = uint32_t(int32_t(uniform_index_word_9) >> 0x2u);

    uniform_parameter_word_20 = uniform_index_word_12 + 0x2u;

    uniform_index_word_15 = uniform_index_word_15 & 0xfffffff0u;

    uniform_index_word_14 = uniform_index_word_14 & 0xfffffff8u;

    if (uniform_condition_0) {
        uniform_address_word_4 =
            uniform_index_word_11 * uniform_parameter_word_21 + uniform_parameter_word_21;
    }

    uniform_index_word_5 = 0u;
    do {
        if (condition_0) {
            matrix_fragment_28 =
                uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                matrix_accumulator_3;

            matrix_accumulator_21 = matrix_accumulator_21 * 0x4u;

            matrix_fragment_13 = matrix_accumulator_2 & 0xfffffffcu;
        }

        split_address_words(matrix_fragment_18, matrix_fragment_19,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));

        if (condition_0) {
            matrix_fragment_14 = parameters[parameter_input_low_0];
            matrix_fragment_15 = parameters[parameter_input_high_1];

            matrix_accumulator_21 =
                matrix_fragment_28 * uniform_index_word_15 + matrix_accumulator_21;
        }

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_18, matrix_fragment_19),
                                            matrix_fragment_36, matrix_fragment_37,
                                            matrix_fragment_38, matrix_fragment_39, memory_ranges,
                                            error_counts);

        if (condition_0) {
            matrix_fragment_13 = matrix_accumulator_21 + address_word_0 + (0u - matrix_fragment_13);
        }

        address_word_0 = matrix_fragment_40;

        matrix_accumulator_2 = address_word_0;

        matrix_fragment_28 = address_word_0;

        matrix_fragment_30 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

        matrix_fragment_22 = uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                             uniform_index_word_12;

        matrix_fragment_12 =
            uint32_t(combine_words(matrix_fragment_30, 0u) << 0x2u >> 32) + matrix_fragment_30;

        if (condition_0) {
            split_address_words(matrix_fragment_14, matrix_fragment_15,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_14, matrix_fragment_15));
        }

        condition_3 = int32_t(matrix_fragment_22) >= int32_t(uniform_index_word_18);

        matrix_accumulator_21 = matrix_fragment_12 & 0xfffffffcu;

        matrix_fragment_12 = uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                             uniform_parameter_word_20;

        matrix_accumulator_21 =
            matrix_fragment_30 + uniform_index_word_13 + (0u - matrix_accumulator_21);

        condition_3 = int32_t(matrix_fragment_22) > -1 && !condition_3;

        condition_1 = int32_t(matrix_fragment_12) >= int32_t(uniform_index_word_18);

        condition_2 = int32_t(matrix_accumulator_21) >= int32_t(uniform_index_word_19);

        condition_1 = int32_t(matrix_fragment_12) > -1 && !condition_1;

        matrix_fragment_12 = 0u;

        condition_2 = int32_t(matrix_accumulator_21) > -1 && !condition_2;

        condition_1 = condition_1 & condition_2;

        if (condition_0) {
            matrix_fragment_12 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_14, matrix_fragment_15), memory_ranges, error_counts);
        }

        condition_0 = condition_3 & condition_2;

        if (condition_1) {
            matrix_fragment_29 = matrix_accumulator_3 + uniform_index_word_18;

            matrix_fragment_22 = parameters[parameter_input_low_0];
            matrix_fragment_23 = parameters[parameter_input_high_1];

            matrix_fragment_13 =
                uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                matrix_accumulator_3;

            matrix_fragment_32 = matrix_accumulator_21 * 0x4u;

            matrix_fragment_33 =
                uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                matrix_fragment_29;

            matrix_fragment_31 = matrix_accumulator_21 * 0x4u;
        }

        if (condition_0) {
            matrix_fragment_29 = matrix_accumulator_3 + uniform_index_word_18;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 + 0x2u;

            matrix_accumulator_26 = parameters[parameter_input_low_0];
            matrix_accumulator_27 = parameters[parameter_input_high_1];

            matrix_fragment_14 = matrix_fragment_33 + 0x2u;
        }

        if (condition_0) {
            matrix_fragment_34 = matrix_accumulator_21 * 0x4u;

            matrix_fragment_29 =
                uint32_t(int64_t(int32_t(matrix_fragment_28)) * (1ll << 0x1cu) >> 32) +
                matrix_fragment_29;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 * uniform_index_word_15 + matrix_fragment_32;

            matrix_fragment_32 = matrix_accumulator_2 & 0xfffffffcu;

            matrix_fragment_15 = matrix_fragment_14 * uniform_index_word_15 + matrix_fragment_31;

            matrix_fragment_14 = matrix_accumulator_2 & 0xfffffffcu;
        }

        if (condition_0) {
            address_word_24 = parameters[parameter_input_low_0];
            address_word_25 = parameters[parameter_input_high_1];

            matrix_fragment_29 = matrix_fragment_29 * uniform_index_word_15 + matrix_fragment_34;

            matrix_fragment_31 = matrix_accumulator_2 & 0xfffffffcu;
        }

        if (condition_1) {
            matrix_fragment_13 = matrix_fragment_13 + address_word_0 + (0u - matrix_fragment_14);

            matrix_fragment_15 = matrix_fragment_15 + address_word_0 + (0u - matrix_fragment_32);
        }

        if (condition_0) {
            matrix_fragment_29 = matrix_fragment_29 + address_word_0 + (0u - matrix_fragment_31);
        }

        if (condition_1) {
            split_address_words(matrix_fragment_22, matrix_fragment_23,
                                int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_fragment_22, matrix_fragment_23));
        }

        matrix_fragment_13 = 0u;

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_18, matrix_fragment_19) - 0x200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        if (condition_1) {
            split_address_words(matrix_accumulator_26, matrix_accumulator_27,
                                int64_t(int32_t(matrix_fragment_15)) * int64_t(int32_t(0x4u)) +
                                    combine_words(matrix_accumulator_26, matrix_accumulator_27));
        }

        matrix_fragment_14 = 0;
        matrix_fragment_15 = 0;

        if (condition_1) {
            matrix_fragment_13 = load_aligned_word<CheckBounds>(
                combine_words(matrix_fragment_22, matrix_fragment_23), memory_ranges, error_counts);

            matrix_fragment_15 = load_aligned_word<CheckBounds>(
                combine_words(matrix_accumulator_26, matrix_accumulator_27), memory_ranges,
                error_counts);
        }

        if (condition_0) {
            split_address_words(address_word_24, address_word_25,
                                int64_t(int32_t(matrix_fragment_29)) * int64_t(int32_t(0x4u)) +
                                    combine_words(address_word_24, address_word_25));

            matrix_fragment_14 = load_aligned_word<CheckBounds>(
                combine_words(address_word_24, address_word_25), memory_ranges, error_counts);
        }

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_10 = uniform_index_word_5 + 0x20u;

        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            combine_words(matrix_fragment_16, matrix_fragment_17) + 0x800u);

        matrix_accumulator_3 = matrix_accumulator_3 + uniform_index_word_14;

        uniform_index_word_5 = uniform_index_word_10;

        multiply_accumulate_half(matrix_fragment_10, matrix_fragment_11, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_32, matrix_fragment_33, matrix_fragment_10,
                                 matrix_fragment_11);

        multiply_accumulate_half(packed_half_4, packed_half_5, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_34, matrix_fragment_35, packed_half_4,
                                 packed_half_5);

        multiply_accumulate_half(matrix_fragment_8, matrix_fragment_9, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                                 matrix_fragment_9);

        multiply_accumulate_half(packed_half_6, matrix_accumulator_7, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_38, matrix_fragment_39, packed_half_6,
                                 matrix_accumulator_7);
    } while (!uniform_condition_0);
    matrix_accumulator_3 = address_word_0 * 0x2u;

    matrix_fragment_12 = uint32_t(uint32_t(address_word_0) >> 0x1u);

    uniform_index_word_14 = parameters[parameter_tensor_height_8];
    uniform_index_word_15 = parameters[parameter_tensor_width_9];

    matrix_fragment_15 = address_word_0;

    matrix_fragment_79 = matrix_accumulator_20 * 0x20u;

    matrix_accumulator_3 = matrix_accumulator_3 & 0x8u;

    uniform_condition_5 = false;

    matrix_fragment_13 = matrix_fragment_12 & 0x4u;

    matrix_fragment_12 = matrix_accumulator_3 | 0x13u & address_word_0;

    matrix_fragment_14 = matrix_fragment_13 | 0x13u & address_word_0;

    matrix_fragment_12 = matrix_fragment_12 | matrix_fragment_13;

    matrix_fragment_13 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1fu);

    matrix_fragment_16 = uint32_t(int32_t(matrix_fragment_15) >> 0x6u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_55 = result_word;
    }

    matrix_fragment_36 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x10u;

    uniform_index_word_10 = uniform_index_word_14 & 0xfffffffcu;

    matrix_fragment_13 =
        uint32_t(combine_words(matrix_fragment_13, 0u) << 0x4u >> 32) + matrix_fragment_30;

    uniform_index_word_5 = uint32_t(int32_t(uniform_index_word_14) >> 0x1fu);

    matrix_fragment_35 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x18u;

    uniform_condition_3 = uint32_t(uniform_index_word_10) != uint32_t(0x4u);

    matrix_fragment_15 =
        uint32_t(combine_words(matrix_fragment_15, 0u) << 0x1u >> 32) + matrix_fragment_16;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_15) >> 0x1fu);

    matrix_accumulator_27 = uint32_t(int32_t(matrix_fragment_36) >> 0x1fu);

    uniform_index_word_5 =
        uint32_t(combine_words(uniform_index_word_5, 0u) << 0x2u >> 32) + uniform_index_word_14;

    matrix_fragment_37 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x8u;

    uniform_index_word_11 =
        uint32_t(combine_words(uniform_index_word_10, 0u) << 0x2u >> 32) + uniform_index_word_15;

    matrix_fragment_13 = matrix_fragment_13 & 0xfff0u;

    uniform_index_word_10 = uint32_t(int32_t(uniform_index_word_5) >> 0x2u);

    matrix_accumulator_26 = uint32_t(int32_t(matrix_fragment_35) >> 0x1fu);

    uniform_index_word_11 = uint32_t(int32_t(uniform_index_word_11) >> 0x2u);

    matrix_fragment_14 = matrix_accumulator_3 | matrix_fragment_14;

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_index_word_8) >= int32_t(uniform_index_word_10);
    }

    matrix_accumulator_3 = matrix_fragment_15 & 0x3ffffffeu;

    uniform_condition_1 = int32_t(uniform_index_word_9) >= int32_t(uniform_index_word_11);

    matrix_fragment_77 =
        uint32_t(combine_words(matrix_accumulator_27, 0u) << 0x4u >> 32) + matrix_fragment_36;

    if (uniform_condition_3) {
        uniform_condition_5 =
            int32_t(uniform_index_word_6) < int32_t(0u - 0x3u) || uniform_condition_0;
    }

    matrix_fragment_18 = uint32_t(int32_t(matrix_fragment_37) >> 0x1fu);

    uniform_index_word_18 = uniform_index_word_15 & 0xfffffffcu;

    matrix_fragment_43 =
        uint32_t(combine_words(matrix_accumulator_26, 0u) << 0x4u >> 32) + matrix_fragment_35;

    uniform_condition_0 = true;

    matrix_fragment_38 = matrix_fragment_30 + (0u - matrix_fragment_13);

    matrix_fragment_13 = matrix_fragment_77 & 0xfff0u;

    uniform_condition_1 =
        int32_t(uniform_index_word_7) > int32_t(0u - 0x4u) && !uniform_condition_1;

    matrix_fragment_33 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x28u;

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_5;
    }

    matrix_accumulator_3 = matrix_fragment_16 + (0u - matrix_accumulator_3);

    matrix_fragment_45 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x4u >> 32) + matrix_fragment_37;

    uniform_condition_2 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    matrix_fragment_16 = matrix_fragment_43 & 0xfff0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    matrix_fragment_34 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x20u;

    uniform_condition_2 = uniform_condition_0 & (uniform_condition_2 | uniform_condition_1);

    matrix_fragment_32 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x30u;

    uniform_condition_4 = uint32_t(uniform_index_word_18) != uint32_t(0x4u);

    matrix_accumulator_27 =
        uint32_t(combine_words(matrix_accumulator_27, 0u) << 0x5u >> 32) + matrix_fragment_36;

    uniform_index_word_12 = uniform_condition_5 ? uniform_index_word_9 : 0u;

    matrix_fragment_28 = uint32_t(int32_t(matrix_fragment_33) >> 0x1fu);

    uniform_index_word_5 = 0u;

    matrix_fragment_36 = matrix_fragment_36 + (0u - matrix_fragment_13);

    matrix_fragment_22 = matrix_fragment_45 & 0xfff0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_8 * uniform_index_word_11;
    }

    matrix_accumulator_26 =
        uint32_t(combine_words(matrix_accumulator_26, 0u) << 0x5u >> 32) + matrix_fragment_35;

    uniform_index_word_12 = !uniform_condition_4 ? uniform_index_word_12 : uniform_index_word_9;

    matrix_fragment_13 = uint32_t(int32_t(matrix_fragment_34) >> 0x1fu);

    if (uniform_condition_2) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_fragment_19 = uint32_t(int32_t(matrix_fragment_32) >> 0x1fu);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_5), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_50 = result_word;
    }

    matrix_fragment_35 = matrix_fragment_35 + (0u - matrix_fragment_16);

    matrix_fragment_16 =
        uint32_t(combine_words(matrix_fragment_28, 0u) << 0x4u >> 32) + matrix_fragment_33;

    if (uniform_condition_2) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    matrix_fragment_29 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x5u >> 32) + matrix_fragment_37;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_69 = result_word;
    }

    matrix_fragment_44 =
        uint32_t(combine_words(matrix_fragment_13, 0u) << 0x4u >> 32) + matrix_fragment_34;

    matrix_fragment_17 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x4u >> 32) + matrix_fragment_32;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_4), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_58 = result_word;
    }

    matrix_fragment_37 = matrix_fragment_37 + (0u - matrix_fragment_22);

    matrix_fragment_22 = matrix_fragment_16 & 0xfff0u;

    matrix_fragment_30 =
        uint32_t(combine_words(matrix_fragment_13, 0u) << 0x5u >> 32) + matrix_fragment_34;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_71 = result_word;
    }

    matrix_fragment_15 = matrix_fragment_44 & 0xfff0u;

    matrix_fragment_13 = matrix_fragment_17 & 0xfff0u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_59 = result_word;
    }

    matrix_fragment_31 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x38u;

    matrix_fragment_28 =
        uint32_t(combine_words(matrix_fragment_28, 0u) << 0x5u >> 32) + matrix_fragment_33;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(matrix_fragment_12), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_72 = result_word;
    }

    matrix_fragment_33 = matrix_fragment_33 + (0u - matrix_fragment_22);

    matrix_fragment_22 =
        uint32_t(combine_words(matrix_fragment_19, 0u) << 0x5u >> 32) + matrix_fragment_32;

    address_word_24 = uint32_t(int32_t(matrix_fragment_31) >> 0x1fu);

    matrix_fragment_34 = matrix_fragment_34 + (0u - matrix_fragment_15);

    matrix_fragment_32 = matrix_fragment_32 + (0u - matrix_fragment_13);

    matrix_fragment_15 = 0x10u ^ matrix_fragment_12;

    matrix_fragment_13 = 0x8u ^ matrix_fragment_14;

    matrix_fragment_14 = 0x18u ^ matrix_fragment_14;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_5), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_41 = result_word;
    }

    matrix_fragment_18 =
        uint32_t(combine_words(address_word_24, 0u) << 0x4u >> 32) + matrix_fragment_31;

    matrix_fragment_23 =
        uint32_t(combine_words(address_word_24, 0u) << 0x5u >> 32) + matrix_fragment_31;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_4), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        address_word_25 = result_word;
    }

    matrix_fragment_40 = matrix_fragment_18 & 0xfff0u;

    condition_1 = (address_word_0 & 0x10u) != 0;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        address_word_24 = result_word;
    }

    condition_0 = uniform_condition_2;

    matrix_fragment_31 = matrix_fragment_31 + (0u - matrix_fragment_40);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_4), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_46 = result_word;
    }

    matrix_fragment_12 = uint32_t(uint32_t(matrix_fragment_79) >> 0x1u);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_21 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_40 = result_word;
    }

    uniform_index_word_12 = uniform_index_word_9 + 0x1u;

    if (uniform_condition_3) {
        uniform_condition_6 = int32_t(uniform_index_word_8) >= int32_t(uniform_index_word_10);
    }

    uniform_condition_5 = false;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_60 = result_word;
    }

    uniform_condition_0 = true;

    matrix_fragment_48 = !condition_1 ? matrix_fragment_50 : matrix_fragment_41;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    matrix_accumulator_61 = !condition_1 ? matrix_fragment_55 : address_word_24;

    matrix_fragment_55 = !condition_1 ? address_word_24 : matrix_fragment_55;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_56 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_19 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_5), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_47 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_73 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_76 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_7), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_80 = result_word;
    }

    matrix_accumulator_62 = !condition_1 ? matrix_accumulator_60 : matrix_fragment_11;

    matrix_accumulator_60 = !condition_1 ? matrix_fragment_11 : matrix_accumulator_60;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_4), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_39 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_65 = result_word;
    }

    matrix_accumulator_7 = !condition_1 ? matrix_fragment_46 : address_word_25;

    packed_half_4 = !condition_1 ? matrix_accumulator_21 : matrix_fragment_42;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_8), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_52 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_74 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_15), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_64 = result_word;
    }

    matrix_fragment_8 = matrix_fragment_12 + 0x4u;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_14), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_49 = result_word;
    }

    if (uniform_condition_3) {
        uniform_condition_5 =
            int32_t(uniform_index_word_6) < int32_t(0u - 0x3u) || uniform_condition_6;
    }

    packed_half_6 = !condition_1 ? matrix_fragment_42 : matrix_accumulator_21;

    matrix_accumulator_21 = matrix_accumulator_2 & 0xfffffffcu;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_5), "r"(matrix_fragment_13), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    matrix_fragment_54 = !condition_1 ? matrix_accumulator_56 : matrix_fragment_19;

    matrix_fragment_15 = matrix_accumulator_2 & 0x7ffffffcu;

    matrix_fragment_11 = matrix_fragment_8 + address_word_0 + (0u - matrix_accumulator_21);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    matrix_accumulator_68 = !condition_1 ? matrix_accumulator_58 : matrix_fragment_39;

    matrix_fragment_51 = !condition_1 ? matrix_fragment_53 : matrix_fragment_40;

    matrix_fragment_50 = !condition_1 ? matrix_fragment_41 : matrix_fragment_50;

    packed_half_5 = !condition_1 ? address_word_25 : matrix_fragment_46;

    matrix_fragment_13 = uniform_index_word_5;

    address_word_24 = parameters[parameter_trained_weights_low_4];
    address_word_25 = parameters[parameter_trained_weights_high_5];

    if (!uniform_condition_2) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_15 = 0u - matrix_fragment_15 + address_word_0;

    matrix_accumulator_56 = !condition_1 ? matrix_fragment_19 : matrix_accumulator_56;

    if (condition_0) {
        matrix_fragment_13 = matrix_fragment_13 * 0x2u + matrix_accumulator_20;
    }

    if (!uniform_condition_2) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_5;
    }

    uniform_condition_6 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_5;

    if (condition_0) {
        matrix_fragment_13 = matrix_fragment_13 * 0x80u;
    }

    uniform_condition_2 = int32_t(uniform_index_word_12) >= int32_t(uniform_index_word_11);

    matrix_accumulator_58 = !condition_1 ? matrix_fragment_39 : matrix_accumulator_58;

    uniform_index_word_13 = uniform_condition_5 ? uniform_index_word_12 : 0u;

    if (condition_0) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    matrix_fragment_13 = matrix_fragment_12 + 0x8u;

    uniform_condition_2 =
        int32_t(uniform_index_word_7) > int32_t(0u - 0x8u) && !uniform_condition_2;

    matrix_accumulator_63 = !condition_1 ? matrix_fragment_10 : matrix_fragment_47;

    matrix_fragment_19 = !condition_1 ? matrix_fragment_47 : matrix_fragment_10;

    matrix_fragment_12 = matrix_fragment_12 + 0xcu;

    matrix_fragment_15 = matrix_fragment_15 * 0x2u + matrix_fragment_79;

    matrix_fragment_13 = matrix_fragment_13 + address_word_0 + (0u - matrix_accumulator_21);

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_6 | uniform_condition_2);

    matrix_fragment_53 = !condition_1 ? matrix_fragment_40 : matrix_fragment_53;

    matrix_accumulator_84 = !condition_1 ? matrix_accumulator_69 : matrix_fragment_76;

    matrix_fragment_88 = !condition_1 ? matrix_fragment_76 : matrix_accumulator_69;

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_11)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_fragment_47 = matrix_fragment_12 + address_word_0 + (0u - matrix_accumulator_21);

    matrix_fragment_87 = uint32_t(int32_t(matrix_fragment_15) >> 0x1u);

    matrix_fragment_39 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x9080ull, memory_ranges,
        error_counts);

    if (!condition_0) {
        matrix_fragment_12 = uniform_index_word_5;

        matrix_fragment_14 = uniform_index_word_5;

        matrix_fragment_15 = uniform_index_word_5;
    }

    if (condition_0) {
        split_address_words(matrix_fragment_8, matrix_fragment_9,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    uniform_index_word_13 = !uniform_condition_4 ? uniform_index_word_13 : uniform_index_word_12;

    if (uniform_condition_0) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_accumulator_57 = !condition_1 ? matrix_accumulator_59 : matrix_accumulator_64;

    split_address_words(matrix_fragment_86, matrix_fragment_87,
                        int64_t(int32_t(matrix_fragment_87)) * int64_t(int32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_accumulator_59 = !condition_1 ? matrix_accumulator_64 : matrix_accumulator_59;

    matrix_fragment_81 = !condition_1 ? matrix_accumulator_65 : matrix_accumulator_74;

    matrix_fragment_83 = !condition_1 ? matrix_accumulator_74 : matrix_accumulator_65;

    split_address_words(matrix_fragment_10, matrix_fragment_11,
                        uint64_t(uint32_t(matrix_fragment_13)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    if (!condition_0) {
        matrix_fragment_13 = uniform_index_word_5;
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_8 * uniform_index_word_11;
    }

    split_address_words(matrix_fragment_46, matrix_fragment_47,
                        uint64_t(uint32_t(matrix_fragment_47)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_10, matrix_fragment_11) + 0x9080ull, memory_ranges,
        error_counts);

    matrix_accumulator_70 = !condition_1 ? matrix_fragment_66 : matrix_fragment_49;

    matrix_accumulator_75 = !condition_1 ? matrix_accumulator_71 : matrix_fragment_52;

    matrix_fragment_78 = !condition_1 ? matrix_fragment_72 : matrix_fragment_67;

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_8, matrix_fragment_9), matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);
    }

    condition_0 = uniform_condition_0;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_13 + uniform_index_word_5;
    }

    if (uniform_condition_3) {
        uniform_index_word_13 = uniform_index_word_8 + 0x1u;
    }

    matrix_fragment_42 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0x9080ull, memory_ranges,
        error_counts);

    uniform_condition_6 = false;

    matrix_accumulator_64 = uint32_t(int32_t(matrix_fragment_43) >> 0x4u);

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_46, matrix_fragment_47) + 0x9080ull, memory_ranges,
        error_counts);

    matrix_fragment_66 = !condition_1 ? matrix_fragment_49 : matrix_fragment_66;

    matrix_fragment_8 = uint32_t(int32_t(matrix_fragment_45) >> 0x4u);

    matrix_accumulator_71 = !condition_1 ? matrix_fragment_52 : matrix_accumulator_71;

    matrix_accumulator_85 = !condition_1 ? matrix_fragment_73 : matrix_fragment_80;

    matrix_fragment_45 =
        uint32_t(combine_words(matrix_fragment_45, 0u) << 0x1u >> 32) + matrix_fragment_8;

    if (uniform_condition_3) {
        uniform_condition_5 = int32_t(uniform_index_word_13) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_9 = uniform_index_word_5;

    matrix_fragment_89 = !condition_1 ? matrix_fragment_80 : matrix_fragment_73;

    matrix_fragment_45 = matrix_fragment_45 & 0x3ffffffeu;

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_index_word_6) < int32_t(0u - 0x7u) || uniform_condition_5;
    }

    matrix_fragment_72 = !condition_1 ? matrix_fragment_67 : matrix_fragment_72;

    matrix_accumulator_69 = matrix_fragment_8 + (0u - matrix_fragment_45);

    if (condition_0) {
        matrix_fragment_45 = matrix_fragment_9 * 0x2u + matrix_accumulator_20;
    }

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    matrix_fragment_47 = uint32_t(int32_t(matrix_fragment_44) >> 0x4u);

    uniform_condition_0 = true;

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (condition_0) {
        matrix_fragment_45 = matrix_fragment_45 * 0x80u;
    }

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_6;
    }

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    matrix_fragment_10 =
        uint32_t(combine_words(matrix_fragment_44, 0u) << 0x1u >> 32) + matrix_fragment_47;

    if (condition_0) {
        split_address_words(matrix_fragment_44, matrix_fragment_45,
                            int64_t(int32_t(matrix_fragment_45)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_8, matrix_fragment_9));
    }

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_1);

    matrix_fragment_43 =
        uint32_t(combine_words(matrix_fragment_43, 0u) << 0x1u >> 32) + matrix_accumulator_64;

    matrix_fragment_46 = matrix_fragment_10 & 0x3ffffffeu;

    if (condition_0) {
        split_address_words(matrix_fragment_44, matrix_fragment_45,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_44, matrix_fragment_45));
    }

    if (!condition_0) {
        matrix_fragment_8 = uniform_index_word_5;

        matrix_fragment_9 = uniform_index_word_5;

        matrix_fragment_10 = uniform_index_word_5;

        matrix_fragment_11 = uniform_index_word_5;
    }

    uniform_index_word_13 = uniform_condition_6 ? uniform_index_word_9 : 0u;

    matrix_fragment_43 = matrix_fragment_43 & 0x3ffffffeu;

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_8 * uniform_index_word_11 + uniform_index_word_11;
    }

    uniform_index_word_13 = !uniform_condition_4 ? uniform_index_word_13 : uniform_index_word_9;

    if (condition_0) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_44, matrix_fragment_45), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }

    if (uniform_condition_0) {
        uniform_parameter_word_20 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_parameter_word_21 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_accumulator_64 = matrix_accumulator_64 + (0u - matrix_fragment_43);

    condition_2 = uniform_condition_0;

    matrix_fragment_43 = matrix_fragment_47 + (0u - matrix_fragment_46);

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_13 + uniform_index_word_5;
    }

    matrix_fragment_47 = uint32_t(int32_t(matrix_fragment_16) >> 0x4u);

    matrix_fragment_46 = uint32_t(int32_t(matrix_fragment_17) >> 0x4u);

    matrix_fragment_16 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x1u >> 32) + matrix_fragment_47;

    matrix_fragment_17 =
        uint32_t(combine_words(matrix_fragment_17, 0u) << 0x1u >> 32) + matrix_fragment_46;

    matrix_accumulator_65 = uniform_index_word_5;

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    if (uniform_condition_3) {
        uniform_index_word_13 = uniform_index_word_8 + 0x1u;
    }

    matrix_fragment_49 = uint32_t(int32_t(matrix_fragment_18) >> 0x4u);

    matrix_fragment_16 = matrix_fragment_16 & 0x3ffffffeu;

    matrix_fragment_17 = matrix_fragment_17 & 0x3ffffffeu;

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    if (condition_2) {
        matrix_accumulator_65 = matrix_accumulator_65 * 0x2u + matrix_accumulator_20;
    }

    if (uniform_condition_3) {
        uniform_condition_0 = int32_t(uniform_index_word_13) >= int32_t(uniform_index_word_10);
    }

    matrix_fragment_18 =
        uint32_t(combine_words(matrix_fragment_18, 0u) << 0x1u >> 32) + matrix_fragment_49;

    uniform_condition_6 = false;

    matrix_fragment_44 = matrix_fragment_47 + (0u - matrix_fragment_16);

    matrix_fragment_46 = matrix_fragment_46 + (0u - matrix_fragment_17);

    split_address_words(matrix_fragment_16, matrix_fragment_17,
                        combine_words(uniform_parameter_word_20, uniform_parameter_word_21));

    if (condition_2) {
        matrix_accumulator_65 = matrix_accumulator_65 * 0x80u;
    }

    if (uniform_condition_3) {
        uniform_condition_6 =
            int32_t(uniform_index_word_6) < int32_t(0u - 0x7u) || uniform_condition_0;
    }

    matrix_fragment_18 = matrix_fragment_18 & 0x3ffffffeu;

    uniform_condition_0 = true;

    condition_0 = (address_word_0 & 0x4u) != 0;

    if (condition_2) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(matrix_accumulator_65)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    if (uniform_condition_3) {
        uniform_condition_0 = !uniform_condition_6;
    }

    uniform_condition_5 = uint32_t(uniform_index_word_18) == uint32_t(0x4u) || uniform_condition_6;

    matrix_fragment_45 = matrix_fragment_49 + (0u - matrix_fragment_18);

    matrix_fragment_47 = !condition_0 ? matrix_fragment_51 : packed_half_4;

    matrix_fragment_49 = !condition_0 ? matrix_fragment_53 : packed_half_6;

    matrix_fragment_76 = !condition_0 ? matrix_accumulator_68 : packed_half_5;

    matrix_accumulator_74 = !condition_0 ? matrix_accumulator_58 : matrix_accumulator_7;

    matrix_fragment_51 = !condition_0 ? packed_half_4 : matrix_fragment_51;

    matrix_fragment_53 = !condition_0 ? packed_half_6 : matrix_fragment_53;

    matrix_accumulator_68 = !condition_0 ? packed_half_5 : matrix_accumulator_68;

    matrix_accumulator_58 = !condition_0 ? matrix_accumulator_7 : matrix_accumulator_58;

    uniform_condition_0 = uniform_condition_0 & (uniform_condition_5 | uniform_condition_2);

    if (!condition_2) {
        packed_half_4 = uniform_index_word_5;

        packed_half_5 = uniform_index_word_5;

        packed_half_6 = uniform_index_word_5;

        matrix_accumulator_7 = uniform_index_word_5;
    }

    if (condition_2) {
        split_address_words(matrix_fragment_16, matrix_fragment_17,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_16, matrix_fragment_17));
    }

    uniform_index_word_5 = 0u;

    if (uniform_condition_3) {
        uniform_index_word_5 = uniform_index_word_8 * uniform_index_word_11 + uniform_index_word_11;
    }

    if (condition_2) {
        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_fragment_16, matrix_fragment_17), packed_half_4, packed_half_5,
            packed_half_6, matrix_accumulator_7, memory_ranges, error_counts);
    }

    if (!uniform_condition_4) {
        uniform_index_word_12 = uniform_condition_6 ? uniform_index_word_12 : 0u;
    }

    if (uniform_condition_0) {
        uniform_index_word_18 = parameters[parameter_input_upsample_skip_optional_low_6];
        uniform_index_word_19 = parameters[parameter_input_upsample_skip_optional_high_7];
    }

    matrix_fragment_52 = uint32_t(int32_t(matrix_fragment_77) >> 0x4u);

    condition_2 = uniform_condition_0;

    if (uniform_condition_0) {
        uniform_index_word_5 = uniform_index_word_12 + uniform_index_word_5;
    }

    matrix_fragment_16 = permute_signed(matrix_fragment_38, 0u, 0x8880u);

    matrix_fragment_17 = permute_signed(matrix_fragment_37, 0u, 0x8880u);

    matrix_fragment_77 =
        uint32_t(combine_words(matrix_fragment_77, 0u) << 0x1u >> 32) + matrix_fragment_52;

    matrix_fragment_16 = permute_signed(matrix_fragment_16, 0u, 0x7710u);

    matrix_fragment_17 = permute_signed(matrix_fragment_17, 0u, 0x7710u);

    matrix_fragment_82 = !condition_0 ? matrix_fragment_78 : matrix_fragment_81;

    matrix_fragment_78 = !condition_0 ? matrix_fragment_81 : matrix_fragment_78;

    matrix_fragment_77 = matrix_fragment_77 & 0x3ffffffeu;

    matrix_fragment_73 = !condition_0 ? matrix_accumulator_61 : matrix_fragment_54;

    matrix_fragment_16 = uint32_t(uint32_t(matrix_fragment_16) >> 0xdu);

    matrix_fragment_81 = uniform_index_word_5;

    matrix_accumulator_61 = !condition_0 ? matrix_fragment_54 : matrix_accumulator_61;

    matrix_fragment_17 = uint32_t(uint32_t(matrix_fragment_17) >> 0xdu);

    matrix_fragment_54 = !condition_0 ? matrix_fragment_50 : matrix_fragment_19;

    matrix_fragment_50 = !condition_0 ? matrix_fragment_19 : matrix_fragment_50;

    matrix_fragment_18 = permute_signed(matrix_fragment_36, 0u, 0x8880u);

    if (condition_2) {
        matrix_fragment_81 = matrix_fragment_81 * 0x2u + matrix_accumulator_20;
    }

    matrix_fragment_19 = matrix_fragment_16 & 0x3u;

    matrix_fragment_52 = matrix_fragment_52 + (0u - matrix_fragment_77);

    matrix_fragment_16 = matrix_fragment_17 & 0x3u;

    matrix_fragment_77 = !condition_0 ? matrix_accumulator_71 : matrix_accumulator_60;

    matrix_accumulator_65 = !condition_0 ? matrix_accumulator_57 : matrix_accumulator_70;

    matrix_accumulator_71 = !condition_0 ? matrix_accumulator_60 : matrix_accumulator_71;

    matrix_accumulator_57 = !condition_0 ? matrix_accumulator_70 : matrix_accumulator_57;

    matrix_fragment_18 = permute_signed(matrix_fragment_18, 0u, 0x7710u);

    if (condition_2) {
        matrix_fragment_17 = matrix_fragment_81 * 0x80u;
    }

    matrix_fragment_80 = !condition_0 ? matrix_fragment_72 : matrix_fragment_83;

    matrix_accumulator_70 = !condition_0 ? matrix_accumulator_84 : matrix_accumulator_85;

    matrix_accumulator_60 = !condition_0 ? matrix_accumulator_85 : matrix_accumulator_84;

    matrix_fragment_72 = !condition_0 ? matrix_fragment_83 : matrix_fragment_72;

    split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                        combine_words(uniform_index_word_18, uniform_index_word_19));

    matrix_fragment_81 = matrix_fragment_38 + matrix_fragment_19;

    matrix_fragment_83 = matrix_fragment_37 + matrix_fragment_16;

    matrix_fragment_18 = uint32_t(uint32_t(matrix_fragment_18) >> 0xdu);

    if (!uniform_condition_0) {
        uniform_index_word_5 = pack_fp8_pair(select_half_operand<2, false, false>(0u),
                                             select_half_operand<2, false, false>(0u));
    }

    matrix_fragment_67 = !condition_0 ? matrix_fragment_55 : matrix_accumulator_56;

    matrix_fragment_55 = !condition_0 ? matrix_accumulator_56 : matrix_fragment_55;

    if (condition_2) {
        split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                            int64_t(int32_t(matrix_fragment_17)) * int64_t(int32_t(0x4u)) +
                                combine_words(matrix_accumulator_84, matrix_accumulator_85));
    }

    matrix_accumulator_56 = !condition_0 ? matrix_fragment_48 : matrix_accumulator_63;

    matrix_fragment_16 = matrix_fragment_81 & 0xfcu;

    if (!uniform_condition_0) {
        uniform_index_word_5 = permute_signed(uniform_index_word_5, uniform_index_word_5, 0x5410u);
    }

    matrix_fragment_17 = matrix_fragment_83 & 0xfcu;

    matrix_fragment_19 = matrix_fragment_18 & 0x3u;

    matrix_fragment_48 = !condition_0 ? matrix_accumulator_63 : matrix_fragment_48;

    matrix_fragment_79 = !condition_0 ? matrix_accumulator_75 : matrix_accumulator_62;

    matrix_accumulator_63 = !condition_0 ? matrix_accumulator_59 : matrix_fragment_66;

    matrix_accumulator_75 = !condition_0 ? matrix_accumulator_62 : matrix_accumulator_75;

    matrix_accumulator_59 = !condition_0 ? matrix_fragment_66 : matrix_accumulator_59;

    matrix_fragment_66 = !condition_0 ? matrix_fragment_88 : matrix_fragment_89;

    matrix_accumulator_62 = !condition_0 ? matrix_fragment_89 : matrix_fragment_88;

    matrix_fragment_89 = 0u - matrix_fragment_16;

    matrix_fragment_90 = 0u - matrix_fragment_17;

    matrix_fragment_88 = matrix_fragment_36 + matrix_fragment_19;

    if (!condition_2) {
        matrix_fragment_16 = uniform_index_word_5;

        matrix_fragment_17 = uniform_index_word_5;

        matrix_fragment_18 = uniform_index_word_5;

        matrix_fragment_19 = uniform_index_word_5;
    }

    if (condition_2) {
        split_address_words(matrix_accumulator_84, matrix_accumulator_85,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_84, matrix_accumulator_85));

        load_four_words_global_cached<CheckBounds>(
            combine_words(matrix_accumulator_84, matrix_accumulator_85), matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);
    }

    matrix_fragment_89 = permute_signed(matrix_fragment_89, 0u, 0x7710u);

    matrix_fragment_90 = permute_signed(matrix_fragment_90, 0u, 0x7710u);

    matrix_fragment_91 = permute_signed(matrix_fragment_35, 0u, 0x8880u);

    matrix_fragment_38 = matrix_fragment_38 + matrix_fragment_89;

    matrix_fragment_37 = matrix_fragment_37 + matrix_fragment_90;

    matrix_fragment_90 = permute_signed(matrix_fragment_91, 0u, 0x7710u);

    matrix_fragment_89 = matrix_fragment_88 & 0xfcu;

    matrix_fragment_91 = permute_signed(matrix_fragment_34, 0u, 0x8880u);

    matrix_accumulator_92 = permute_signed(matrix_fragment_33, 0u, 0x8880u);

    matrix_fragment_90 = uint32_t(uint32_t(matrix_fragment_90) >> 0xdu);

    matrix_accumulator_93 = 0u - matrix_fragment_89;

    matrix_fragment_89 = permute_signed(matrix_fragment_91, 0u, 0x7710u);

    matrix_accumulator_94 = permute_signed(matrix_fragment_32, 0u, 0x8880u);

    matrix_fragment_91 = permute_signed(matrix_accumulator_92, 0u, 0x7710u);

    matrix_fragment_90 = matrix_fragment_90 & 0x3u;

    matrix_fragment_89 = uint32_t(uint32_t(matrix_fragment_89) >> 0xdu);

    matrix_accumulator_92 = permute_signed(matrix_accumulator_94, 0u, 0x7710u);

    matrix_fragment_91 = uint32_t(uint32_t(matrix_fragment_91) >> 0xdu);

    matrix_fragment_90 = matrix_fragment_35 + matrix_fragment_90;

    matrix_accumulator_93 = permute_signed(matrix_accumulator_93, 0u, 0x7710u);

    matrix_fragment_89 = matrix_fragment_89 & 0x3u;

    matrix_accumulator_85 = uint32_t(uint32_t(matrix_accumulator_92) >> 0xdu);

    matrix_accumulator_92 = matrix_fragment_91 & 0x3u;

    matrix_accumulator_84 = matrix_fragment_90 & 0xfcu;

    matrix_fragment_36 = matrix_fragment_36 + matrix_accumulator_93;

    matrix_accumulator_85 = matrix_accumulator_85 & 0x3u;

    matrix_fragment_89 = matrix_fragment_34 + matrix_fragment_89;

    matrix_accumulator_93 = permute_signed(matrix_fragment_31, 0u, 0x8880u);

    matrix_accumulator_92 = matrix_fragment_33 + matrix_accumulator_92;

    matrix_fragment_96 = 0u - matrix_accumulator_84;

    matrix_accumulator_93 = permute_signed(matrix_accumulator_93, 0u, 0x7710u);

    matrix_fragment_91 = matrix_fragment_32 + matrix_accumulator_85;

    matrix_accumulator_84 = matrix_fragment_89 & 0xfcu;

    matrix_accumulator_85 = matrix_accumulator_92 & 0xfcu;

    matrix_fragment_96 = permute_signed(matrix_fragment_96, 0u, 0x7710u);

    matrix_accumulator_94 = uint32_t(uint32_t(matrix_accumulator_93) >> 0xdu);

    matrix_accumulator_93 = matrix_fragment_91 & 0xfcu;

    matrix_accumulator_84 = 0u - matrix_accumulator_84;

    matrix_accumulator_95 = 0u - matrix_accumulator_85;

    matrix_accumulator_94 = matrix_accumulator_94 & 0x3u;

    matrix_fragment_35 = matrix_fragment_35 + matrix_fragment_96;

    matrix_fragment_96 = 0u - matrix_accumulator_93;

    matrix_accumulator_85 = permute_signed(matrix_accumulator_84, 0u, 0x7710u);

    matrix_accumulator_93 = permute_signed(matrix_fragment_81, 0u, 0x8880u);

    matrix_accumulator_84 = permute_signed(matrix_accumulator_95, 0u, 0x7710u);

    matrix_accumulator_94 = matrix_fragment_31 + matrix_accumulator_94;

    matrix_fragment_81 = permute_signed(matrix_fragment_96, 0u, 0x7710u);

    matrix_fragment_34 = matrix_fragment_34 + matrix_accumulator_85;

    matrix_accumulator_85 = matrix_accumulator_93;

    matrix_fragment_33 = matrix_fragment_33 + matrix_accumulator_84;

    matrix_accumulator_93 = permute_signed(matrix_fragment_83, 0u, 0x8880u);

    matrix_accumulator_84 = matrix_accumulator_94 & 0xfcu;

    matrix_fragment_32 = matrix_fragment_32 + matrix_fragment_81;

    matrix_fragment_81 = address_word_0;

    matrix_fragment_83 = uint32_t(int32_t(matrix_accumulator_85) >> 0x2u);

    matrix_accumulator_85 = matrix_accumulator_93;

    matrix_accumulator_84 = 0u - matrix_accumulator_84;

    matrix_fragment_81 = uint32_t(int32_t(matrix_fragment_81) >> 0x7u);

    matrix_accumulator_93 = permute_signed(matrix_fragment_83, 0u, 0x9910u);

    matrix_fragment_88 = permute_signed(matrix_fragment_88, 0u, 0x8880u);

    matrix_fragment_83 = uint32_t(int32_t(matrix_accumulator_85) >> 0x2u);

    matrix_fragment_96 = permute_signed(matrix_accumulator_84, 0u, 0x7710u);

    matrix_accumulator_84 = (matrix_fragment_81 << 0x2u) + uniform_index_word_6;

    matrix_fragment_81 = matrix_accumulator_93;

    matrix_accumulator_85 = uint32_t(int32_t(matrix_fragment_88) >> 0x2u);

    matrix_fragment_88 = permute_signed(matrix_fragment_83, 0u, 0x9910u);

    matrix_fragment_29 = uint32_t(int32_t(matrix_fragment_29) >> 0x5u);

    matrix_fragment_83 = uint32_t(int32_t(matrix_accumulator_27) >> 0x5u);

    matrix_fragment_81 = matrix_fragment_81 + matrix_accumulator_84;

    matrix_accumulator_84 = (matrix_fragment_29 << 0x2u) + uniform_index_word_6;

    matrix_accumulator_27 = matrix_fragment_88;

    matrix_fragment_29 = permute_signed(matrix_accumulator_85, 0u, 0x9910u);

    matrix_fragment_88 = (matrix_fragment_83 << 0x2u) + uniform_index_word_6;

    matrix_fragment_90 = permute_signed(matrix_fragment_90, 0u, 0x8880u);

    matrix_fragment_88 = matrix_fragment_29 + matrix_fragment_88;

    matrix_fragment_29 = matrix_fragment_90;

    matrix_accumulator_27 = matrix_accumulator_27 + matrix_accumulator_84;

    matrix_fragment_29 = uint32_t(int32_t(matrix_fragment_29) >> 0x2u);

    matrix_accumulator_26 = uint32_t(int32_t(matrix_accumulator_26) >> 0x5u);

    matrix_accumulator_85 = permute_signed(matrix_fragment_29, 0u, 0x9910u);

    matrix_fragment_83 = permute_signed(matrix_fragment_89, 0u, 0x8880u);

    matrix_accumulator_84 = permute_signed(matrix_accumulator_92, 0u, 0x8880u);

    condition_3 = int32_t(matrix_fragment_81) >= int32_t(0u);

    condition_4 = int32_t(matrix_fragment_81) >= int32_t(uniform_index_word_14);

    matrix_fragment_30 = uint32_t(int32_t(matrix_fragment_30) >> 0x5u);

    matrix_fragment_81 = uint32_t(int32_t(matrix_fragment_28) >> 0x5u);

    matrix_fragment_29 = (matrix_accumulator_26 << 0x2u) + uniform_index_word_6;

    matrix_fragment_28 = uint32_t(int32_t(matrix_fragment_83) >> 0x2u);

    matrix_accumulator_84 = uint32_t(int32_t(matrix_accumulator_84) >> 0x2u);

    matrix_accumulator_26 = matrix_accumulator_85;

    matrix_fragment_91 = permute_signed(matrix_fragment_91, 0u, 0x8880u);

    matrix_fragment_83 = (matrix_fragment_30 << 0x2u) + uniform_index_word_6;

    matrix_fragment_28 = permute_signed(matrix_fragment_28, 0u, 0x9910u);

    matrix_fragment_30 = permute_signed(matrix_accumulator_84, 0u, 0x9910u);

    matrix_accumulator_85 = (matrix_fragment_81 << 0x2u) + uniform_index_word_6;

    matrix_fragment_81 = matrix_accumulator_26 + matrix_fragment_29;

    matrix_accumulator_94 = permute_signed(matrix_accumulator_94, 0u, 0x8880u);

    matrix_fragment_29 = matrix_fragment_91;

    matrix_fragment_38 = matrix_fragment_38 & 0xffffu;

    matrix_fragment_28 = matrix_fragment_28 + matrix_fragment_83;

    matrix_accumulator_26 = matrix_fragment_30 + matrix_accumulator_85;

    matrix_fragment_22 = uint32_t(int32_t(matrix_fragment_22) >> 0x5u);

    matrix_fragment_83 = matrix_accumulator_94;

    matrix_fragment_30 = uint32_t(int32_t(matrix_fragment_23) >> 0x5u);

    matrix_fragment_23 = uint32_t(int32_t(matrix_fragment_29) >> 0x2u);

    matrix_fragment_38 = permute_signed(matrix_fragment_38, 0u, 0x8880u);

    matrix_fragment_29 = (matrix_fragment_22 << 0x2u) + uniform_index_word_6;

    matrix_fragment_22 = uint32_t(int32_t(matrix_fragment_83) >> 0x2u);

    matrix_fragment_23 = permute_signed(matrix_fragment_23, 0u, 0x9910u);

    matrix_accumulator_84 = (matrix_fragment_30 << 0x2u) + uniform_index_word_6;

    matrix_fragment_30 = (matrix_accumulator_3 << 0x2u) + uniform_index_word_7;

    matrix_fragment_37 = matrix_fragment_37 & 0xffffu;

    matrix_accumulator_3 = matrix_fragment_38;

    matrix_fragment_83 = permute_signed(matrix_fragment_22, 0u, 0x9910u);

    matrix_fragment_22 = matrix_fragment_23;

    matrix_fragment_23 = permute_signed(matrix_fragment_37, 0u, 0x8880u);

    matrix_fragment_38 = (matrix_accumulator_69 << 0x2u) + uniform_index_word_7;

    matrix_fragment_30 = matrix_accumulator_3 + matrix_fragment_30;

    condition_2 = int32_t(matrix_accumulator_27) >= int32_t(0u);

    condition_0 = int32_t(matrix_accumulator_27) >= int32_t(uniform_index_word_14);

    matrix_accumulator_27 = matrix_fragment_23 + matrix_fragment_38;

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    condition_5 = int32_t(matrix_fragment_30) >= int32_t(uniform_index_word_15);

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_23 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_79));

    condition_6 = int32_t(matrix_fragment_30) < int32_t(0u) || condition_5;

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_fragment_14 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_37 = matrix_fragment_83;

    matrix_fragment_23 = !condition_6 ? matrix_fragment_23 : 0u;

    matrix_fragment_12 = !condition_6 ? matrix_fragment_12 : 0u;

    matrix_fragment_38 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_76));

    matrix_fragment_22 = matrix_fragment_22 + matrix_fragment_29;

    matrix_fragment_29 = !condition_6 ? matrix_fragment_14 : 0u;

    matrix_fragment_14 = !condition_4 ? matrix_fragment_23 : 0u;

    matrix_accumulator_3 = matrix_fragment_37 + matrix_accumulator_84;

    matrix_fragment_23 = !condition_4 ? matrix_fragment_12 : 0u;

    matrix_fragment_38 = !condition_6 ? matrix_fragment_38 : 0u;

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_15);

    condition_6 = int32_t(matrix_accumulator_27) >= int32_t(uniform_index_word_15);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_36 = matrix_fragment_36 & 0xffffu;

    matrix_fragment_37 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_accumulator_74));

    condition_6 = int32_t(matrix_accumulator_27) < int32_t(0u) || condition_6;

    matrix_fragment_77 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_fragment_77));

    matrix_accumulator_27 = permute_signed(matrix_fragment_36, 0u, 0x8880u);

    matrix_fragment_80 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_52 = (matrix_fragment_52 << 0x2u) + uniform_index_word_7;

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_37 = !condition_6 ? matrix_fragment_37 : 0u;

    matrix_fragment_77 = !condition_6 ? matrix_fragment_77 : 0u;

    matrix_fragment_80 = !condition_6 ? matrix_fragment_80 : 0u;

    matrix_fragment_52 = matrix_accumulator_27 + matrix_fragment_52;

    matrix_fragment_67 = !condition_6 ? matrix_fragment_67 : 0u;

    matrix_fragment_13 = condition_3 ? matrix_fragment_23 : 0u;

    matrix_fragment_37 = !condition_0 ? matrix_fragment_37 : 0u;

    matrix_fragment_77 = !condition_0 ? matrix_fragment_77 : 0u;

    matrix_fragment_80 = !condition_0 ? matrix_fragment_80 : 0u;

    matrix_fragment_67 = !condition_0 ? matrix_fragment_67 : 0u;

    matrix_fragment_23 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_accumulator_27 = unpack_fp8_bits(matrix_fragment_10);

    condition_0 = int32_t(matrix_fragment_52) >= int32_t(uniform_index_word_15);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_35 = matrix_fragment_35 & 0xffffu;

    matrix_accumulator_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_23),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_accumulator_68));

    condition_0 = int32_t(matrix_fragment_52) < int32_t(0u) || condition_0;

    matrix_accumulator_75 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_30 = permute_signed(matrix_fragment_35, 0u, 0x8880u);

    matrix_fragment_78 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_78));

    matrix_accumulator_61 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_35 = (matrix_accumulator_64 << 0x2u) + uniform_index_word_7;

    matrix_fragment_38 = !condition_4 ? matrix_fragment_38 : 0u;

    matrix_fragment_29 = !condition_4 ? matrix_fragment_29 : 0u;

    condition_4 = int32_t(matrix_fragment_88) >= int32_t(uniform_index_word_14);

    matrix_accumulator_68 = !condition_0 ? matrix_accumulator_68 : 0u;

    matrix_accumulator_75 = !condition_0 ? matrix_accumulator_75 : 0u;

    matrix_accumulator_61 = !condition_0 ? matrix_accumulator_61 : 0u;

    matrix_fragment_78 = !condition_0 ? matrix_fragment_78 : 0u;

    matrix_fragment_30 = matrix_fragment_30 + matrix_fragment_35;

    matrix_fragment_8 = condition_2 ? matrix_fragment_37 : 0u;

    matrix_fragment_23 = condition_2 ? matrix_fragment_77 : 0u;

    matrix_fragment_10 = condition_2 ? matrix_fragment_67 : 0u;

    matrix_accumulator_27 = condition_2 ? matrix_fragment_80 : 0u;

    matrix_accumulator_68 = !condition_4 ? matrix_accumulator_68 : 0u;

    matrix_accumulator_75 = !condition_4 ? matrix_accumulator_75 : 0u;

    matrix_accumulator_61 = !condition_4 ? matrix_accumulator_61 : 0u;

    matrix_fragment_78 = !condition_4 ? matrix_fragment_78 : 0u;

    matrix_fragment_15 = condition_3 ? matrix_fragment_29 : 0u;

    condition_2 = int32_t(matrix_fragment_28) >= int32_t(0u);

    condition_4 = int32_t(matrix_fragment_28) >= int32_t(uniform_index_word_14);

    condition_0 = int32_t(matrix_fragment_30) >= int32_t(uniform_index_word_15);

    matrix_fragment_29 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_28 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_34 = matrix_fragment_34 & 0xffffu;

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_71));

    condition_0 = int32_t(matrix_fragment_30) < int32_t(0u) || condition_0;

    matrix_fragment_29 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_30 = permute_signed(matrix_fragment_34, 0u, 0x8880u);

    matrix_fragment_36 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_fragment_43 = (matrix_fragment_43 << 0x2u) + uniform_index_word_7;

    matrix_fragment_55 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_55));

    condition_6 = int32_t(matrix_fragment_81) >= int32_t(uniform_index_word_14);

    matrix_fragment_28 = !condition_0 ? matrix_fragment_28 : 0u;

    condition_5 = int32_t(matrix_fragment_88) >= int32_t(0u);

    matrix_fragment_29 = !condition_0 ? matrix_fragment_29 : 0u;

    matrix_fragment_55 = !condition_0 ? matrix_fragment_55 : 0u;

    matrix_fragment_36 = !condition_0 ? matrix_fragment_36 : 0u;

    matrix_fragment_43 = matrix_fragment_30 + matrix_fragment_43;

    matrix_fragment_30 = !condition_6 ? matrix_fragment_28 : 0u;

    matrix_fragment_11 = condition_5 ? matrix_accumulator_68 : 0u;

    matrix_accumulator_75 = condition_5 ? matrix_accumulator_75 : 0u;

    matrix_fragment_9 = condition_5 ? matrix_accumulator_61 : 0u;

    matrix_fragment_78 = condition_5 ? matrix_fragment_78 : 0u;

    matrix_fragment_29 = !condition_6 ? matrix_fragment_29 : 0u;

    matrix_fragment_55 = !condition_6 ? matrix_fragment_55 : 0u;

    matrix_fragment_36 = !condition_6 ? matrix_fragment_36 : 0u;

    matrix_fragment_28 = unpack_fp8_bits(packed_half_4 >> 16);

    condition_5 = int32_t(matrix_accumulator_26) >= int32_t(0u);

    condition_0 = int32_t(matrix_accumulator_26) >= int32_t(uniform_index_word_14);

    condition_6 = int32_t(matrix_fragment_43) >= int32_t(uniform_index_word_15);

    packed_half_4 = unpack_fp8_bits(packed_half_4);

    matrix_accumulator_26 = unpack_fp8_bits(packed_half_6);

    matrix_fragment_35 = unpack_fp8_bits(packed_half_6 >> 16);

    matrix_fragment_31 = matrix_fragment_31 + matrix_fragment_96;

    matrix_fragment_33 = matrix_fragment_33 & 0xffffu;

    matrix_fragment_32 = matrix_fragment_32 & 0xffffu;

    packed_half_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    condition_6 = int32_t(matrix_fragment_43) < int32_t(0u) || condition_6;

    matrix_fragment_28 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_12 = condition_3 ? matrix_fragment_38 : 0u;

    matrix_accumulator_26 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_fragment_14 = condition_3 ? matrix_fragment_14 : 0u;

    matrix_fragment_35 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_70));

    matrix_fragment_37 = (matrix_fragment_44 << 0x2u) + uniform_index_word_7;

    matrix_fragment_31 = matrix_fragment_31 & 0xffffu;

    packed_half_6 = permute_signed(matrix_fragment_33, 0u, 0x8880u);

    matrix_fragment_32 = permute_signed(matrix_fragment_32, 0u, 0x8880u);

    condition_3 = int32_t(matrix_fragment_81) >= int32_t(0u);

    packed_half_4 = !condition_6 ? packed_half_4 : 0u;

    matrix_fragment_33 = permute_signed(matrix_fragment_31, 0u, 0x8880u);

    matrix_fragment_28 = !condition_6 ? matrix_fragment_28 : 0u;

    matrix_accumulator_26 = !condition_6 ? matrix_accumulator_26 : 0u;

    matrix_fragment_35 = !condition_6 ? matrix_fragment_35 : 0u;

    matrix_fragment_31 = matrix_fragment_32;

    matrix_fragment_37 = packed_half_6 + matrix_fragment_37;

    matrix_fragment_29 = condition_3 ? matrix_fragment_29 : 0u;

    matrix_fragment_32 = condition_3 ? matrix_fragment_30 : 0u;

    matrix_fragment_55 = condition_3 ? matrix_fragment_55 : 0u;

    matrix_fragment_36 = condition_3 ? matrix_fragment_36 : 0u;

    condition_3 = int32_t(matrix_fragment_22) >= int32_t(0u);

    condition_6 = int32_t(matrix_fragment_22) >= int32_t(uniform_index_word_14);

    matrix_fragment_22 = !condition_4 ? packed_half_4 : 0u;

    matrix_fragment_30 = matrix_fragment_33;

    matrix_fragment_28 = !condition_4 ? matrix_fragment_28 : 0u;

    matrix_accumulator_26 = !condition_4 ? matrix_accumulator_26 : 0u;

    matrix_fragment_35 = !condition_4 ? matrix_fragment_35 : 0u;

    packed_half_4 = unpack_fp8_bits(packed_half_5 >> 16);

    condition_4 = int32_t(matrix_fragment_37) >= int32_t(uniform_index_word_15);

    packed_half_6 = unpack_fp8_bits(matrix_accumulator_7);

    matrix_fragment_33 = unpack_fp8_bits(matrix_accumulator_7 >> 16);

    packed_half_5 = unpack_fp8_bits(packed_half_5);

    packed_half_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_54));

    condition_4 = int32_t(matrix_fragment_37) < int32_t(0u) || condition_4;

    packed_half_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_fragment_33 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    packed_half_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_5),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_49));

    matrix_fragment_45 = (matrix_fragment_45 << 0x2u) + uniform_index_word_7;

    packed_half_4 = !condition_4 ? packed_half_4 : 0u;

    packed_half_6 = !condition_4 ? packed_half_6 : 0u;

    packed_half_5 = !condition_4 ? packed_half_5 : 0u;

    matrix_fragment_33 = !condition_4 ? matrix_fragment_33 : 0u;

    matrix_fragment_46 = (matrix_fragment_46 << 0x2u) + uniform_index_word_7;

    matrix_fragment_45 = matrix_fragment_30 + matrix_fragment_45;

    packed_half_4 = !condition_0 ? packed_half_4 : 0u;

    packed_half_6 = !condition_0 ? packed_half_6 : 0u;

    packed_half_5 = !condition_0 ? packed_half_5 : 0u;

    matrix_fragment_30 = !condition_0 ? matrix_fragment_33 : 0u;

    matrix_fragment_31 = matrix_fragment_31 + matrix_fragment_46;

    matrix_fragment_34 = condition_2 ? matrix_accumulator_26 : 0u;

    matrix_accumulator_7 = condition_2 ? matrix_fragment_35 : 0u;

    matrix_fragment_28 = condition_2 ? matrix_fragment_28 : 0u;

    matrix_fragment_22 = condition_2 ? matrix_fragment_22 : 0u;

    matrix_fragment_33 = condition_5 ? packed_half_4 : 0u;

    matrix_fragment_35 = condition_5 ? packed_half_6 : 0u;

    matrix_accumulator_26 = condition_5 ? packed_half_5 : 0u;

    matrix_fragment_37 = condition_5 ? matrix_fragment_30 : 0u;

    condition_2 = int32_t(matrix_accumulator_3) >= int32_t(0u);

    condition_5 = int32_t(matrix_accumulator_3) >= int32_t(uniform_index_word_14);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_16);

    uniform_index_word_6 = 0u;

    condition_4 = int32_t(matrix_fragment_31) >= int32_t(uniform_index_word_15);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    packed_half_4 = unpack_fp8_bits(matrix_fragment_17);

    condition_4 = int32_t(matrix_fragment_31) < int32_t(0u) || condition_4;

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_48));

    condition_0 = int32_t(matrix_fragment_45) >= int32_t(uniform_index_word_15);

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_30 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    matrix_fragment_31 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_accumulator_3 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_51));

    packed_half_5 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_42 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                                      select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_53));

    packed_half_6 = unpack_fp8_bits(matrix_fragment_19);

    matrix_fragment_39 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                      select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(matrix_fragment_50));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_fragment_16 = !condition_4 ? matrix_fragment_16 : 0u;

    condition_0 = int32_t(matrix_fragment_45) < int32_t(0u) || condition_0;

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_41),
                                      select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_57 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_5),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_accumulator_3 = !condition_4 ? matrix_accumulator_3 : 0u;

    matrix_fragment_40 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                                      select_half_operand<2, false, false>(matrix_fragment_40),
                                      select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                       select_half_operand<2, false, false>(packed_half_4));

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = !condition_6 ? matrix_fragment_16 : 0u;

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = !condition_4 ? matrix_fragment_30 : 0u;

    matrix_fragment_39 = !condition_0 ? matrix_fragment_39 : 0u;

    matrix_fragment_31 = !condition_0 ? matrix_fragment_31 : 0u;

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                         select_half_operand<2, false, false>(packed_half_5));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = !condition_6 ? matrix_accumulator_3 : 0u;

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(packed_half_4));

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = condition_3 ? matrix_fragment_16 : 0u;

    matrix_accumulator_57 = !condition_4 ? matrix_accumulator_57 : 0u;

    matrix_fragment_42 = !condition_0 ? matrix_fragment_42 : 0u;

    matrix_fragment_40 = !condition_0 ? matrix_fragment_40 : 0u;

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                  select_half_operand<2, false, false>(0u));

    uniform_index_word_5 = 0x400u;

    matrix_fragment_30 = !condition_6 ? matrix_fragment_30 : 0u;

    matrix_fragment_39 = !condition_5 ? matrix_fragment_39 : 0u;

    matrix_fragment_31 = !condition_5 ? matrix_fragment_31 : 0u;

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    uniform_index_word_6 = (uniform_index_word_6 << 0x18u) + uniform_index_word_5;

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(packed_half_5));

    matrix_accumulator_3 = condition_3 ? matrix_accumulator_3 : 0u;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                  select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_57 = !condition_6 ? matrix_accumulator_57 : 0u;

    matrix_fragment_42 = !condition_5 ? matrix_fragment_42 : 0u;

    matrix_fragment_40 = !condition_5 ? matrix_fragment_40 : 0u;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = condition_3 ? matrix_fragment_30 : 0u;

    matrix_fragment_39 = condition_2 ? matrix_fragment_39 : 0u;

    matrix_fragment_31 = condition_2 ? matrix_fragment_31 : 0u;

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                  select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_3),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_3 = (matrix_accumulator_20 << 0x9u) + uniform_index_word_6;

    matrix_accumulator_57 = condition_3 ? matrix_accumulator_57 : 0u;

    matrix_fragment_42 = condition_2 ? matrix_fragment_42 : 0u;

    matrix_fragment_40 = condition_2 ? matrix_fragment_40 : 0u;

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_3 = address_word_0 * 0x10u + matrix_accumulator_3;

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                  select_half_operand<2, false, false>(packed_half_6));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    uniform_index_word_12 = parameters[parameter_trained_weights_low_4];
    uniform_index_word_13 = parameters[parameter_trained_weights_high_5];

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3 + 0x400u, packed_half_4,
                                         packed_half_5, packed_half_6, matrix_accumulator_7,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x800u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0xc00u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_22 = 0;
    matrix_fragment_23 = 0;

    matrix_accumulator_26 = 0;
    matrix_accumulator_27 = 0;

    matrix_accumulator_58 = 0;
    matrix_accumulator_59 = 0;

    matrix_accumulator_68 = 0;
    matrix_accumulator_69 = 0;

    matrix_accumulator_56 = 0;
    matrix_accumulator_57 = 0;

    matrix_accumulator_60 = 0;
    matrix_accumulator_61 = 0;

    matrix_accumulator_62 = 0;
    matrix_accumulator_63 = 0;

    matrix_accumulator_70 = 0;
    matrix_accumulator_71 = 0;

    matrix_fragment_78 = 0;
    matrix_fragment_79 = 0;

    matrix_accumulator_64 = 0;
    matrix_accumulator_65 = 0;

    matrix_fragment_82 = 0;
    matrix_fragment_83 = 0;

    matrix_fragment_66 = 0;
    matrix_fragment_67 = 0;

    matrix_fragment_80 = 0;
    matrix_fragment_81 = 0;

    matrix_accumulator_74 = 0;
    matrix_accumulator_75 = 0;

    matrix_fragment_76 = 0;
    matrix_fragment_77 = 0;

    matrix_accumulator_84 = 0;
    matrix_accumulator_85 = 0;

    split_address_words(matrix_fragment_72, matrix_fragment_73,
                        combine_words(uniform_index_word_12, uniform_index_word_13));

    uniform_index_word_5 = 0u;

    __syncthreads();
    do {
        split_address_words(packed_half_4, packed_half_5,
                            uint64_t(uint32_t(matrix_accumulator_20)) *
                                    uint64_t(uint32_t(0x2000u)) +
                                combine_words(matrix_fragment_72, matrix_fragment_73));

        split_address_words(matrix_fragment_28, matrix_fragment_29,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_4, packed_half_5));

        load_four_words_cached<CheckBounds>(combine_words(matrix_fragment_28, matrix_fragment_29),
                                            matrix_fragment_52, matrix_fragment_53,
                                            matrix_fragment_54, matrix_fragment_55, memory_ranges,
                                            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x200ull, matrix_fragment_32,
            matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x1000ull, matrix_fragment_16,
            matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges,
            error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_fragment_28, matrix_fragment_29) + 0x1200ull, matrix_fragment_12,
            matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges,
            error_counts);

        matrix_fragment_98 = (address_word_0 << 0x4u) + uniform_index_word_6;

        split_address_words(packed_half_4, packed_half_5,
                            uint64_t(uint32_t(matrix_accumulator_20)) *
                                    uint64_t(uint32_t(0x1000u)) +
                                combine_words(matrix_fragment_72, matrix_fragment_73));

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0x400u, matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

        split_address_words(matrix_accumulator_94, matrix_accumulator_95,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(packed_half_4, packed_half_5));

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x4000ull, packed_half_4,
            packed_half_5, packed_half_6, matrix_accumulator_7, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0x800u, matrix_fragment_88, matrix_fragment_89,
            matrix_fragment_90, matrix_fragment_91, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0xc00u, matrix_fragment_48, matrix_fragment_49,
            matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0x200u, matrix_fragment_44, matrix_fragment_45,
            matrix_fragment_46, matrix_fragment_47, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0x600u, matrix_fragment_40, matrix_fragment_41,
            matrix_fragment_42, matrix_fragment_43, shared_bytes, memory_ranges, error_counts);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0xa00u, matrix_fragment_28, matrix_fragment_29,
            matrix_fragment_30, matrix_fragment_31, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_32, matrix_fragment_33, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_8,
                                 matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                 matrix_fragment_34, matrix_fragment_35, 0u, 0u);

        load_four_words_cached<CheckBounds>(
            combine_words(matrix_accumulator_94, matrix_accumulator_95) + 0x4200ull,
            matrix_fragment_8, matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
            memory_ranges, error_counts);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_32, matrix_fragment_33, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_34, matrix_fragment_35, 0u, 0u);

        load_four_shared_words<CheckBounds>(
            shared_tile, matrix_fragment_98 + 0xe00u, matrix_fragment_36, matrix_fragment_37,
            matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

        multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(packed_half_118, matrix_accumulator_119, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_54, matrix_fragment_55, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_16, matrix_fragment_17, matrix_fragment_96,
                                 matrix_fragment_97);

        multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_100,
                                 matrix_accumulator_101);

        multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_52, matrix_fragment_53, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_32, matrix_fragment_33, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_88,
                                 matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                                 matrix_fragment_34, matrix_fragment_35, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_106,
                                 matrix_accumulator_107);

        multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_108,
                                 matrix_accumulator_109);

        matrix_fragment_99 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_96), 0x44004400u);

        matrix_fragment_98 = 0x2b28u;

        matrix_fragment_99 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99), 0xc400c400u);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_32, matrix_fragment_33, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_34, matrix_fragment_35, 0u, 0u);

        multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_112,
                                 matrix_accumulator_113);

        multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_18, matrix_fragment_19, matrix_accumulator_114,
                                 matrix_accumulator_115);

        multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_18, matrix_fragment_19, packed_half_118,
                                 matrix_accumulator_119);

        multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_102,
                                 matrix_accumulator_103);

        matrix_accumulator_103 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_100), 0x44004400u);

        matrix_accumulator_102 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_97), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                                 matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_104,
                                 matrix_accumulator_105);

        matrix_accumulator_105 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103), 0xc400c400u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_102), 0xc400c400u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_99),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_accumulator_104 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_101), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_12, matrix_fragment_13, matrix_accumulator_110,
                                 matrix_accumulator_111);

        matrix_accumulator_107 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_104), 0xc400c400u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99),
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_40,
                                 matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                 matrix_fragment_14, matrix_fragment_15, matrix_accumulator_92,
                                 matrix_accumulator_93);

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_90), 0x44004400u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_94), 0x44004400u);

        matrix_fragment_41 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_91), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_16, matrix_fragment_17, matrix_accumulator_116,
                                 matrix_accumulator_117);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_99 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_95), 0x44004400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x3b283b28u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_48), 0x44004400u);

        matrix_fragment_96 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_99),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_54), 0x44004400u);

        matrix_fragment_90 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_41 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_49), 0x44004400u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99),
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

        matrix_fragment_99 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_fragment_42 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41), 0xc400c400u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_41), 0x3b283b28u);

        matrix_fragment_40 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16), 0x44004400u);

        multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_52,
                                 matrix_fragment_53);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40), 0xc400c400u);

        matrix_fragment_48 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_88,
                                 matrix_fragment_89);

        matrix_fragment_28 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0x44004400u);

        matrix_fragment_29 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_17), 0x44004400u);

        matrix_fragment_31 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_28), 0xc400c400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_30 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51), 0x44004400u);

        matrix_fragment_29 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29), 0xc400c400u);

        matrix_fragment_28 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_28), 0x3b283b28u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_31),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_41 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_30), 0xc400c400u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_29),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_12, matrix_fragment_13, matrix_fragment_32,
                                 matrix_fragment_33);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_31),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        matrix_fragment_28 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                select_half_operand<2, false, false>(matrix_fragment_28));

        matrix_fragment_16 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_44), 0x44004400u);

        matrix_fragment_30 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_29),
            select_half_operand<2, false, false>(matrix_fragment_30), 0x3b283b28u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_18), 0x44004400u);

        matrix_fragment_13 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_19), 0x44004400u);

        matrix_fragment_29 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16), 0xc400c400u);

        matrix_fragment_16 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_13), 0xc400c400u);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_36,
                                 matrix_fragment_37, matrix_fragment_38, matrix_fragment_39,
                                 matrix_fragment_14, matrix_fragment_15, matrix_fragment_34,
                                 matrix_fragment_35);

        matrix_fragment_13 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_13), 0x3b283b28u);

        matrix_fragment_12 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_45), 0x44004400u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_16),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_92), 0x44004400u);

        matrix_fragment_16 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_16),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_46), 0x44004400u);

        matrix_fragment_30 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                select_half_operand<2, false, false>(matrix_fragment_30));

        matrix_fragment_12 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12), 0xc400c400u);

        matrix_fragment_17 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_47), 0x44004400u);

        matrix_fragment_97 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                select_half_operand<2, false, false>(matrix_accumulator_104));

        matrix_fragment_15 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_93), 0x44004400u);

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_91 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_55), 0x44004400u);

        matrix_fragment_17 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_17), 0x3b283b28u);

        matrix_fragment_37 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_12),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_40 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_15),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x3b283b28u);

        matrix_accumulator_103 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43), 0xc400c400u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_42),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_12 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_12),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_36 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15),
            select_half_operand<2, false, false>(matrix_fragment_36), 0x3b283b28u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_37),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_39 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(matrix_fragment_39), 0x3b283b28u);

        matrix_fragment_38 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_52), 0x44004400u);

        matrix_fragment_15 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_40),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_43 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42),
            select_half_operand<2, false, false>(matrix_fragment_43), 0x3b283b28u);

        matrix_fragment_38 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38), 0xc400c400u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_41),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_37 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_14 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_88), 0x44004400u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_40),
            select_half_operand<2, false, false>(matrix_fragment_15), 0x3b283b28u);

        matrix_fragment_15 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_53), 0x44004400u);

        matrix_accumulator_106 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_105),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_31 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_fragment_42 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_89), 0x44004400u);

        matrix_accumulator_108 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_107),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_41 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_14), 0xc400c400u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_38),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_15 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15), 0xc400c400u);

        matrix_accumulator_106 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_105),
            select_half_operand<2, false, false>(matrix_accumulator_106), 0x3b283b28u);

        matrix_fragment_49 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_43 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_42), 0xc400c400u);

        matrix_accumulator_108 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_107),
            select_half_operand<2, false, false>(matrix_accumulator_108), 0x3b283b28u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_38),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_41),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_14 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_15),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_accumulator_100 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                select_half_operand<2, false, false>(matrix_accumulator_106));

        matrix_fragment_17 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_accumulator_101 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                select_half_operand<2, false, false>(matrix_accumulator_108));

        matrix_accumulator_94 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_94),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_fragment_42 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_41),
            select_half_operand<2, false, false>(matrix_fragment_42), 0x3b283b28u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_99),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_41 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_15),
            select_half_operand<2, false, false>(matrix_fragment_14), 0x3b283b28u);

        matrix_fragment_12 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                                select_half_operand<2, false, false>(matrix_fragment_12));

        matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_14 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_100),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_95 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_95),
                                select_half_operand<2, false, false>(matrix_accumulator_104));

        matrix_fragment_50 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_32), 0x44004400u);

        matrix_accumulator_102 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_99),
            select_half_operand<2, false, false>(matrix_accumulator_102), 0x3b283b28u);

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_101),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_39 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_92),
                                select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_93),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_12 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                          select_half_operand<2, false, false>(matrix_fragment_13));

        matrix_fragment_16 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                select_half_operand<2, false, false>(matrix_fragment_16));

        matrix_fragment_14 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_31 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_51 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_50), 0xc400c400u);

        matrix_fragment_54 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_54),
                                select_half_operand<2, false, false>(matrix_accumulator_102));

        matrix_fragment_13 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_43),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_36 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                                select_half_operand<2, false, false>(matrix_fragment_36));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_37 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                                select_half_operand<2, false, false>(matrix_fragment_37));

        matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_15 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                          select_half_operand<2, false, false>(matrix_fragment_15));

        matrix_fragment_50 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_43),
            select_half_operand<2, false, false>(matrix_fragment_50), 0x3b283b28u);

        matrix_fragment_16 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                          select_half_operand<2, false, false>(matrix_fragment_17));

        matrix_fragment_42 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_17 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                          select_half_operand<2, false, false>(matrix_fragment_18));

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_18 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                          select_half_operand<2, false, false>(matrix_fragment_39));

        matrix_fragment_19 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                          select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_37 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_34), 0x44004400u);

        matrix_fragment_39 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_35), 0x44004400u);

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_48 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_36 = minimum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_33), 0x44004400u);

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_37 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37), 0xc400c400u);

        matrix_fragment_39 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39), 0xc400c400u);

        matrix_fragment_89 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                                select_half_operand<2, false, false>(matrix_fragment_50));

        matrix_fragment_36 = maximum_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36), 0xc400c400u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_accumulator_103),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_50 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                          select_half_operand<2, false, false>(matrix_fragment_43));

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_37),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_39),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_51),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_accumulator_104 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_accumulator_103),
            select_half_operand<2, false, false>(matrix_accumulator_104), 0x3b283b28u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, true, false>(matrix_fragment_36),
            select_half_operand<0, false, true>(matrix_fragment_98), 0x37283728u);

        matrix_fragment_38 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_37),
            select_half_operand<2, false, false>(matrix_fragment_38), 0x3b283b28u);

        matrix_fragment_40 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_39),
            select_half_operand<2, false, false>(matrix_fragment_40), 0x3b283b28u);

        matrix_fragment_55 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_55),
                                select_half_operand<2, false, false>(matrix_accumulator_104));

        matrix_fragment_99 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_51),
            select_half_operand<2, false, false>(matrix_fragment_99), 0x3b283b28u);

        matrix_fragment_98 = fused_multiply_add_half_pairs(
            select_half_operand<2, false, false>(matrix_fragment_36),
            select_half_operand<2, false, false>(matrix_fragment_98), 0x3b283b28u);

        matrix_fragment_38 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                                select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_40 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                                select_half_operand<2, false, false>(matrix_fragment_40));

        matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_41 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_53),
                                select_half_operand<2, false, false>(matrix_fragment_41));

        matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_99 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                select_half_operand<2, false, false>(matrix_fragment_99));

        matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_98 =
            multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                select_half_operand<2, false, false>(matrix_fragment_98));

        matrix_fragment_42 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                           select_half_operand<2, false, false>(0u));

        matrix_fragment_28 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                          select_half_operand<2, false, false>(matrix_fragment_29));

        matrix_fragment_29 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                          select_half_operand<2, false, false>(matrix_fragment_31));

        matrix_fragment_49 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                          select_half_operand<2, false, false>(matrix_fragment_44));

        matrix_fragment_51 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                          select_half_operand<2, false, false>(matrix_fragment_42));

        matrix_fragment_30 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                          select_half_operand<2, false, false>(matrix_fragment_38));

        matrix_fragment_31 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_98),
                          select_half_operand<2, false, false>(matrix_fragment_33));

        multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 packed_half_4, packed_half_5, matrix_accumulator_84,
                                 matrix_accumulator_85);

        uniform_condition_0 = uint32_t(uniform_index_word_5) >= uint32_t(0x60u);

        uniform_index_word_7 = uniform_index_word_5 + 0x20u;

        multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 packed_half_4, packed_half_5, matrix_fragment_66,
                                 matrix_fragment_67);

        multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_4, packed_half_5, matrix_accumulator_70,
                                 matrix_accumulator_71);

        multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 packed_half_4, packed_half_5, matrix_accumulator_68,
                                 matrix_accumulator_69);

        multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 packed_half_6, matrix_accumulator_7, matrix_fragment_76,
                                 matrix_fragment_77);

        multiply_accumulate_half(matrix_fragment_82, matrix_fragment_83, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 packed_half_6, matrix_accumulator_7, matrix_fragment_82,
                                 matrix_fragment_83);

        multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 packed_half_6, matrix_accumulator_7, matrix_accumulator_62,
                                 matrix_accumulator_63);

        multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 packed_half_6, matrix_accumulator_7, matrix_accumulator_58,
                                 matrix_accumulator_59);

        multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_74,
                                 matrix_accumulator_75);

        multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_12,
                                 matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_80,
                                 matrix_fragment_81);

        multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_64,
                                 matrix_accumulator_65);

        multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_16,
                                 matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_78,
                                 matrix_fragment_79);

        multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_60,
                                 matrix_accumulator_61);

        multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_48,
                                 matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                 matrix_fragment_10, matrix_fragment_11, matrix_accumulator_56,
                                 matrix_accumulator_57);

        multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_8, matrix_fragment_9, matrix_accumulator_26,
                                 matrix_accumulator_27);

        multiply_accumulate_half(matrix_fragment_22, matrix_fragment_23, matrix_fragment_28,
                                 matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                 matrix_fragment_10, matrix_fragment_11, matrix_fragment_22,
                                 matrix_fragment_23);

        split_address_words(matrix_fragment_72, matrix_fragment_73,
                            combine_words(matrix_fragment_72, matrix_fragment_73) + 0x400u);

        uniform_index_word_5 = uniform_index_word_7;
    } while (!uniform_condition_0);
    matrix_fragment_120 = threadIdx.y;

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_77),
                                  select_half_operand<2, false, false>(0u));

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                  select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_73 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                       select_half_operand<2, false, false>(packed_half_4));

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_76),
                                  select_half_operand<2, false, false>(0u));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x400u, matrix_fragment_8, matrix_fragment_9,
        matrix_fragment_10, matrix_fragment_11, shared_bytes, memory_ranges, error_counts);

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_83),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_75 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                      select_half_operand<2, false, false>(packed_half_6));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x800u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_72 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                       select_half_operand<2, false, false>(packed_half_5));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0xc00u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_79),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_49 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_130 = threadIdx.y;

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_82),
                                  select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_74 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                      select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                       select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_78),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_48 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(packed_half_5));

    matrix_fragment_120 = matrix_fragment_120 * 0x20u;

    matrix_accumulator_65 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_71),
                      select_half_operand<2, false, false>(packed_half_4));

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                                  select_half_operand<2, false, false>(0u));

    index_word_159 = uint32_t(uint32_t(matrix_fragment_120) >> 0x1u);

    matrix_accumulator_62 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0x9000ull, memory_ranges,
        error_counts);

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                                       select_half_operand<2, false, false>(matrix_accumulator_7));

    reused_word_165 = index_word_159 + 0x4u;

    reused_word_164 = index_word_159 + 0x8u;

    index_word_159 = index_word_159 + 0xcu;

    matrix_accumulator_64 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_70),
                      select_half_operand<2, false, false>(packed_half_5));

    address_word_161 = reused_word_165 + address_word_0 + (0u - matrix_accumulator_21);

    address_word_153 = reused_word_164 + address_word_0 + (0u - matrix_accumulator_21);

    address_word_163 = index_word_159 + address_word_0 + (0u - matrix_accumulator_21);

    split_address_words(packed_half_160, address_word_161,
                        uint64_t(uint32_t(address_word_161)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_69 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_69),
                      select_half_operand<2, false, false>(packed_half_4));

    split_address_words(packed_half_152, address_word_153,
                        uint64_t(uint32_t(address_word_153)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_accumulator_59 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, address_word_161) + 0x9000ull, memory_ranges, error_counts);

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                      select_half_operand<2, false, false>(0u));

    split_address_words(packed_half_162, address_word_163,
                        uint64_t(uint32_t(address_word_163)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_accumulator_58 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_152, address_word_153) + 0x9000ull, memory_ranges, error_counts);

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                                       select_half_operand<2, false, false>(matrix_accumulator_7));

    packed_half_4 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_162, address_word_163) + 0x9000ull, memory_ranges, error_counts);

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                       select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_22),
                                         select_half_operand<2, false, false>(0u));

    __syncthreads();

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_23),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_68 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_68),
                      select_half_operand<2, false, false>(packed_half_5));

    matrix_accumulator_70 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                      select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_71 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(packed_half_6));

    matrix_fragment_23 = matrix_accumulator_130 * 0x100u;

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3, matrix_fragment_72, matrix_fragment_73,
        matrix_accumulator_74, matrix_accumulator_75, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x400u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x800u, matrix_accumulator_64, matrix_accumulator_65,
        matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0xc00u, matrix_accumulator_68, matrix_accumulator_69,
        matrix_accumulator_70, matrix_accumulator_71, shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_22, matrix_fragment_23,
                        uint64_t(uint32_t(matrix_fragment_23)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    split_address_words(matrix_fragment_22, matrix_fragment_23,
                        int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(matrix_fragment_22, matrix_fragment_23));

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x6000ull, matrix_fragment_40,
        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x6200ull, matrix_fragment_36,
        matrix_fragment_37, matrix_fragment_38, matrix_fragment_39, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x6800ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x6a00ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    packed_half_5 = (address_word_0 << 0x4u) + uniform_index_word_6;

    condition_0 = (address_word_0 & 0x1u) != 0;

    condition_2 = (address_word_0 & 0x2u) != 0;

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5, matrix_fragment_52,
                                        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                                        shared_bytes, memory_ranges, error_counts);

    packed_half_6 = unpack_fp8_bits(matrix_fragment_44);

    condition_3 = (address_word_0 & 0x8u) != 0;

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x400u, matrix_fragment_48,
                                        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_7 = unpack_fp8_bits(matrix_fragment_45);

    uniform_index_word_5 = blockIdx.y;

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_44 >> 16);

    uniform_index_word_6 = parameters[parameter_optional_control_11];

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_45 >> 16);

    matrix_accumulator_26 = unpack_fp8_bits(matrix_fragment_46);

    matrix_accumulator_27 = unpack_fp8_bits(matrix_fragment_47);

    matrix_fragment_46 = unpack_fp8_bits(matrix_fragment_46 >> 16);

    matrix_fragment_47 = unpack_fp8_bits(matrix_fragment_47 >> 16);

    packed_half_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_accumulator_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_46 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(packed_half_4));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    multiply_accumulate_half(packed_half_6, matrix_accumulator_7, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_40, matrix_fragment_41, packed_half_6,
                             matrix_accumulator_7);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_36, matrix_fragment_37, matrix_accumulator_26,
                             matrix_accumulator_27);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_52,
                             matrix_fragment_53, matrix_fragment_54, matrix_fragment_55,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_44,
                             matrix_fragment_45);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x800u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_13);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0xc00u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_14);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_17);

    matrix_fragment_16 = unpack_fp8_bits(matrix_fragment_16 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    matrix_fragment_17 = unpack_fp8_bits(matrix_fragment_17 >> 16);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_62));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_40, matrix_fragment_41, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_18);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_19);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_42, matrix_fragment_43, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_18 = unpack_fp8_bits(matrix_fragment_18 >> 16);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x600u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_19 = unpack_fp8_bits(matrix_fragment_19 >> 16);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x200u, matrix_fragment_12,
                                        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(packed_half_4));

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0xa00u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_36,
                             matrix_fragment_37);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0xe00u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(packed_half_6, matrix_accumulator_7, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_32, matrix_fragment_33, packed_half_6,
                             matrix_accumulator_7);

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_56,
                             matrix_accumulator_57);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_48,
                             matrix_fragment_49);

    multiply_accumulate_half(matrix_fragment_52, matrix_fragment_53, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_52,
                             matrix_fragment_53);

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_54,
                             matrix_fragment_55);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_44,
                             matrix_fragment_45);

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                  select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_50,
                             matrix_fragment_51);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_58,
                             matrix_accumulator_59);

    __syncthreads();

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_6),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                                      select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                      select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_52),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_53),
                                       select_half_operand<2, false, false>(0u));

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_64),
                      select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_65),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                                       select_half_operand<2, false, false>(matrix_fragment_12));

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                                       select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_64 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                      select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_65 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                          select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_66 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_66),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                      select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_67 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_67),
                                       select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                      select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_accumulator_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                          select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                      select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_59 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(matrix_fragment_12));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3, matrix_fragment_8,
                                         matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                         shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x400u, matrix_fragment_16, matrix_fragment_17,
        matrix_fragment_18, matrix_fragment_19, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x800u, matrix_accumulator_64, matrix_accumulator_65,
        matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0xc00u, matrix_accumulator_56, matrix_accumulator_57,
        matrix_accumulator_58, matrix_accumulator_59, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    matrix_fragment_15 = matrix_accumulator_130 * 0x300u;

    split_address_words(packed_half_6, matrix_accumulator_7,
                        uint64_t(uint32_t(matrix_fragment_15)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    split_address_words(packed_half_6, matrix_accumulator_7,
                        int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(packed_half_6, matrix_accumulator_7));

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x9300ull, matrix_fragment_48,
        matrix_fragment_49, matrix_fragment_50, matrix_fragment_51, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x9100ull, matrix_fragment_44,
        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x9500ull, matrix_fragment_32,
        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x9700ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x9900ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x9b00ull, matrix_fragment_72,
        matrix_fragment_73, matrix_accumulator_74, matrix_accumulator_75, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0xab00ull, matrix_accumulator_68,
        matrix_accumulator_69, matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0xa900ull, matrix_accumulator_60,
        matrix_accumulator_61, matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0xb100ull, matrix_fragment_52,
        matrix_fragment_53, matrix_fragment_54, matrix_fragment_55, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0xad00ull, matrix_fragment_144,
        matrix_fragment_145, matrix_fragment_146, matrix_fragment_147, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0xaf00ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0xb300ull, matrix_accumulator_56,
        matrix_accumulator_57, matrix_accumulator_58, matrix_accumulator_59, memory_ranges,
        error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5, matrix_fragment_28,
                                        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x400u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x200u, matrix_fragment_96,
                                        matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_126, matrix_fragment_127, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_28,
                             matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x600u, matrix_fragment_16,
                                        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_104,
                             matrix_accumulator_105);

    multiply_accumulate_half(matrix_fragment_126, matrix_fragment_127, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_126,
                             matrix_fragment_127);

    multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_120,
                             matrix_fragment_121);

    multiply_accumulate_half(packed_half_6, matrix_accumulator_7, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_40,
                             matrix_fragment_41);

    packed_half_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_accumulator_104));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_92,
                             matrix_accumulator_93);

    packed_half_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                                      select_half_operand<2, false, false>(matrix_fragment_120),
                                      select_half_operand<2, false, false>(packed_half_4));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_144, matrix_fragment_145, matrix_accumulator_110,
                             matrix_accumulator_111);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_146, matrix_fragment_147, matrix_fragment_42,
                             matrix_fragment_43);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x800u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_108,
                             matrix_accumulator_109);

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_112,
                             matrix_accumulator_113);

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_94,
                             matrix_accumulator_95);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_96,
                             matrix_fragment_97, matrix_fragment_98, matrix_fragment_99,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_64,
                             matrix_accumulator_65);

    load_four_shared_words<CheckBounds>(
        shared_tile, packed_half_5 + 0xc00u, matrix_accumulator_64, matrix_accumulator_65,
        matrix_fragment_66, matrix_fragment_67, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_134,
                             matrix_accumulator_135);

    multiply_accumulate_half(matrix_accumulator_138, matrix_accumulator_139, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_138,
                             matrix_accumulator_139);

    multiply_accumulate_half(matrix_accumulator_136, matrix_accumulator_137, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_accumulator_136,
                             matrix_accumulator_137);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_144, matrix_fragment_145, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_146, matrix_fragment_147, matrix_accumulator_100,
                             matrix_accumulator_101);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_accumulator_124, matrix_accumulator_125, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_124,
                             matrix_accumulator_125);

    multiply_accumulate_half(matrix_fragment_98, matrix_fragment_99, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_fragment_54, matrix_fragment_55, matrix_fragment_76,
                             matrix_fragment_77);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0xa00u, matrix_fragment_76,
                                        matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_38,
                             matrix_fragment_39);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_16,
                             matrix_fragment_17, matrix_fragment_18, matrix_fragment_19,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_84,
                             matrix_accumulator_85);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_126),
                            select_half_operand<2, false, false>(matrix_fragment_126));

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                                      select_half_operand<2, false, false>(packed_half_6),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_17 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                        select_half_operand<2, false, false>(packed_half_4));

    packed_half_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_127),
                            select_half_operand<2, false, false>(matrix_fragment_127));

    packed_half_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                                      select_half_operand<2, false, false>(matrix_fragment_121),
                                      select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_16 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                                      select_half_operand<2, false, false>(matrix_accumulator_7),
                                      select_half_operand<2, false, false>(matrix_fragment_16));

    packed_half_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                   select_half_operand<2, false, false>(packed_half_4));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_4), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_81 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_106, matrix_accumulator_107, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_32, matrix_fragment_33, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_17), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_32 = result_word;
    }

    matrix_fragment_81 = add_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_81), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_80 = result_word;
    }

    packed_half_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_accumulator_108));

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                        select_half_operand<2, false, false>(matrix_fragment_32));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_32), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_33 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    packed_half_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(matrix_accumulator_110),
                                      select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_30),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_83 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                        select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                        select_half_operand<2, false, false>(matrix_fragment_80));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_83), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_114 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_156, matrix_accumulator_157, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    matrix_fragment_81 = permute_signed(matrix_fragment_80, matrix_fragment_80, 0x5432u);

    matrix_fragment_80 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                                        select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(matrix_accumulator_109));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_accumulator_111),
                                      select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_81 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_31),
                                      select_half_operand<2, false, false>(matrix_fragment_81));

    packed_half_4 = permute_signed(matrix_fragment_33, matrix_fragment_33, 0x5432u);

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                                     select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_fragment_33 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                        select_half_operand<2, false, false>(packed_half_4));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_118), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_142,
                             matrix_accumulator_143);

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_33 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33), 0x4100410u);

    matrix_fragment_80 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_83), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_116 = result_word;
    }

    packed_half_4 = half_to_float_bits((matrix_fragment_33 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_accumulator_156, matrix_accumulator_157, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_146, matrix_fragment_147, matrix_accumulator_156,
                             matrix_accumulator_157);

    matrix_fragment_33 = half_to_float_bits((matrix_fragment_33 >> 16) * 65537u);

    matrix_fragment_81 = half_to_float_bits((matrix_fragment_80 & 65535u) * 65537u);

    matrix_fragment_80 = half_to_float_bits((matrix_fragment_80 >> 16) * 65537u);

    matrix_fragment_32 = rsqrt_bits(matrix_fragment_33);

    matrix_fragment_82 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_accumulator_117 = rsqrt_bits(packed_half_4);

    matrix_fragment_82 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_accumulator_100),
                                      select_half_operand<2, false, false>(matrix_fragment_82));

    matrix_accumulator_114 = rsqrt_bits(matrix_fragment_81);

    packed_half_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_accumulator_26));

    multiply_accumulate_half(matrix_fragment_18, matrix_fragment_19, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_144, matrix_fragment_145, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_33 = rsqrt_bits(matrix_fragment_80);

    packed_half_4 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(matrix_fragment_28),
                                      select_half_operand<2, false, false>(packed_half_4));

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                                     select_half_operand<2, false, false>(matrix_accumulator_115));

    packed_half_4 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                                   select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_142));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_118), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_129 = result_word;
    }

    matrix_accumulator_119 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_115 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_156),
                                      select_half_operand<2, false, false>(matrix_accumulator_156),
                                      select_half_operand<2, false, false>(matrix_fragment_80));

    multiply_accumulate_half(matrix_accumulator_148, matrix_accumulator_149, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_34, matrix_fragment_35, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_4), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_35 = result_word;
    }

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0xe00u, matrix_fragment_80,
                                        matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_117 = pack_half(matrix_fragment_32, matrix_accumulator_117);

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_accumulator_114 = pack_half(matrix_fragment_33, matrix_accumulator_114);

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_32 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_33 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_accumulator_101),
                                      select_half_operand<2, false, false>(matrix_fragment_33));

    matrix_fragment_34 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_18),
                                      select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    packed_half_6 = permute_signed(matrix_accumulator_119, matrix_accumulator_119, 0x5432u);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                       select_half_operand<2, false, false>(matrix_fragment_34));

    matrix_fragment_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_126),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_12, matrix_fragment_13, 0u, 0u);

    matrix_accumulator_119 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                       select_half_operand<2, false, false>(packed_half_6));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                                     select_half_operand<2, false, false>(matrix_accumulator_129));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_115), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_6 = result_word;
    }

    split_address_words(matrix_fragment_12, matrix_fragment_13,
                        uint64_t(uint32_t(matrix_accumulator_130)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                       select_half_operand<2, false, false>(matrix_fragment_35));

    packed_half_4 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_12, matrix_fragment_13) + 0x10100ull, memory_ranges,
        error_counts);

    matrix_accumulator_119 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x4100410u);

    multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_14, matrix_fragment_15, 0u, 0u);

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    __syncthreads();

    matrix_fragment_13 = permute_signed(packed_half_118, packed_half_118, 0x5432u);

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                                     select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_fragment_15 = half_to_float_bits((matrix_accumulator_119 >> 16) * 65537u);

    packed_half_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_118), 0x4100410u);

    matrix_accumulator_128 = half_to_float_bits((matrix_accumulator_119 & 65535u) * 65537u);

    multiply_accumulate_half(matrix_fragment_144, matrix_fragment_145, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_144, matrix_fragment_145, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_15 = rsqrt_bits(matrix_fragment_15);

    matrix_accumulator_117 = half_to_float_bits((packed_half_118 & 65535u) * 65537u);

    packed_half_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_115),
                                   select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_115 = matrix_accumulator_130 * 0x800u;

    matrix_accumulator_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_99 = transpose_half_tile(matrix_fragment_99);

    matrix_accumulator_107 = half_to_float_bits((packed_half_118 >> 16) * 65537u);

    matrix_fragment_14 = rsqrt_bits(matrix_accumulator_128);

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_105),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_125 = transpose_half_tile(matrix_accumulator_125);

    matrix_fragment_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_127),
                            select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_17));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_116), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_105 = result_word;
    }

    matrix_accumulator_114 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_accumulator_143));

    matrix_accumulator_106 = rsqrt_bits(matrix_accumulator_117);

    split_address_words(address_word_24, address_word_25,
                        uint64_t(uint32_t(matrix_accumulator_115)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_fragment_12 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_19),
                                      select_half_operand<2, false, false>(matrix_fragment_12));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    matrix_accumulator_114 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_157),
                                      select_half_operand<2, false, false>(matrix_accumulator_157),
                                      select_half_operand<2, false, false>(matrix_accumulator_114));

    multiply_accumulate_half(matrix_accumulator_148, matrix_accumulator_149, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_146, matrix_fragment_147, matrix_accumulator_148,
                             matrix_accumulator_149);

    matrix_fragment_36 = transpose_half_tile(matrix_fragment_36);

    matrix_accumulator_107 = rsqrt_bits(matrix_accumulator_107);

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_fragment_12));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_104), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_13 = result_word;
    }

    matrix_fragment_14 = pack_half(matrix_fragment_15, matrix_fragment_14);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_117 = result_word;
    }

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_accumulator_93 = transpose_half_tile(matrix_accumulator_93);

    multiply_accumulate_half(matrix_fragment_146, matrix_fragment_147, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_32,
                             matrix_fragment_33);

    matrix_accumulator_92 = transpose_half_tile(matrix_accumulator_92);

    packed_half_140 = pack_half(matrix_accumulator_107, matrix_accumulator_106);

    matrix_accumulator_95 = transpose_half_tile(matrix_accumulator_95);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_accumulator_138));

    matrix_accumulator_133 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_fragment_97 = transpose_half_tile(matrix_fragment_97);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_31 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_94 = transpose_half_tile(matrix_accumulator_94);

    matrix_fragment_30 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_accumulator_134),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_32 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                       select_half_operand<2, false, false>(matrix_accumulator_115));

    multiply_accumulate_half(matrix_accumulator_150, matrix_accumulator_151, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_150,
                             matrix_accumulator_151);

    packed_half_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                                   select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = permute_signed(matrix_accumulator_116, matrix_accumulator_116, 0x5432u);

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_104),
                                     select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_160 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_160, address_word_161) + 0x11110ull, memory_ranges, error_counts);

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_116),
                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_31 = permute_signed(matrix_fragment_32, matrix_fragment_32, 0x5432u);

    split_address_words(address_word_24, address_word_25,
                        int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                            combine_words(address_word_24, address_word_25));

    matrix_accumulator_119 = permute_signed(packed_half_118, packed_half_118, 0x5432u);

    packed_half_152 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_152, address_word_153) + 0x11110ull, memory_ranges, error_counts);

    matrix_fragment_32 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_114), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_31 = result_word;
    }

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                                     select_half_operand<2, false, false>(matrix_accumulator_119));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xc100ull,
                                        matrix_accumulator_104, matrix_accumulator_105,
                                        matrix_accumulator_106, matrix_accumulator_107,
                                        memory_ranges, error_counts);

    packed_half_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_118), 0x4100410u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_111),
                            select_half_operand<2, false, false>(packed_half_140));

    packed_half_162 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_162, address_word_163) + 0x11110ull, memory_ranges, error_counts);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_109),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_accumulator_116 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x4100410u);

    packed_half_140 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(packed_half_140));

    matrix_fragment_30 = half_to_float_bits((packed_half_118 & 65535u) * 65537u);

    matrix_accumulator_113 = half_to_float_bits((packed_half_118 >> 16) * 65537u);

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_accumulator_139));

    matrix_fragment_33 = half_to_float_bits((matrix_accumulator_116 & 65535u) * 65537u);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_30 = rsqrt_bits(matrix_fragment_30);

    matrix_accumulator_116 = half_to_float_bits((matrix_accumulator_116 >> 16) * 65537u);

    matrix_fragment_10 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_accumulator_135),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_113 = rsqrt_bits(matrix_accumulator_113);

    matrix_fragment_31 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_110),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_33 = rsqrt_bits(matrix_fragment_33);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_108),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_fragment_10 = permute_signed(matrix_fragment_31, matrix_fragment_31, 0x5432u);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xc300ull,
                                        matrix_accumulator_108, matrix_accumulator_109,
                                        matrix_accumulator_110, matrix_accumulator_111,
                                        memory_ranges, error_counts);

    matrix_accumulator_116 = rsqrt_bits(matrix_accumulator_116);

    matrix_fragment_32 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32), 0x4100410u);

    matrix_fragment_31 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                                        select_half_operand<2, false, false>(matrix_fragment_10));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        packed_half_158 = result_word;
    }

    matrix_accumulator_117 = half_to_float_bits((matrix_fragment_32 & 65535u) * 65537u);

    matrix_accumulator_119 = half_to_float_bits((matrix_fragment_32 >> 16) * 65537u);

    matrix_fragment_32 = pack_half(matrix_accumulator_113, matrix_fragment_30);

    packed_half_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31), 0x4100410u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_115 = result_word;
    }

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_117 = rsqrt_bits(matrix_accumulator_117);

    matrix_accumulator_26 = half_to_float_bits((packed_half_118 & 65535u) * 65537u);

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_116 = pack_half(matrix_accumulator_116, matrix_fragment_33);

    matrix_fragment_28 = half_to_float_bits((packed_half_118 >> 16) * 65537u);

    matrix_accumulator_128 = rsqrt_bits(matrix_accumulator_119);

    matrix_fragment_33 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_26 = rsqrt_bits(matrix_accumulator_26);

    matrix_fragment_29 = rsqrt_bits(matrix_fragment_28);

    packed_half_158 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                     select_half_operand<2, false, false>(packed_half_158));

    matrix_accumulator_117 = pack_half(matrix_accumulator_128, matrix_accumulator_117);

    matrix_fragment_10 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                       select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_131 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_128 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_18 = packed_half_158;

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_101 = result_word;
    }

    matrix_fragment_29 = pack_half(matrix_fragment_29, matrix_accumulator_26);

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(matrix_fragment_29));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_18), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    matrix_accumulator_133 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_fragment_14));

    matrix_accumulator_130 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xc500ull,
                                        matrix_accumulator_112, matrix_accumulator_113,
                                        matrix_accumulator_114, matrix_accumulator_115,
                                        memory_ranges, error_counts);

    matrix_accumulator_132 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_accumulator_129 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_156),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_127 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_accumulator_142 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_142),
                            select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_28 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                      select_half_operand<2, false, false>(matrix_accumulator_133));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xc700ull,
                                        matrix_accumulator_116, matrix_accumulator_117,
                                        packed_half_118, matrix_accumulator_119, memory_ranges,
                                        error_counts);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_146),
                            select_half_operand<2, false, false>(matrix_fragment_146));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                            select_half_operand<2, false, false>(matrix_accumulator_150));

    packed_half_140 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_140),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_144),
                                      select_half_operand<2, false, false>(matrix_fragment_144),
                                      select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_11 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_148),
                                      select_half_operand<2, false, false>(matrix_accumulator_148),
                                      select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_156 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_accumulator_157 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_157),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    packed_half_155 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_143),
                            select_half_operand<2, false, false>(matrix_fragment_29));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                       select_half_operand<2, false, false>(packed_half_140));

    packed_half_140 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                     select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_16 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                                        select_half_operand<2, false, false>(matrix_fragment_8));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xcb00ull,
                                        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10,
                                        matrix_fragment_11, memory_ranges, error_counts);

    packed_half_158 = add_half_pairs(select_half_operand<2, false, false>(packed_half_158),
                                     select_half_operand<2, false, false>(matrix_fragment_17));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_16), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_17 = result_word;
    }

    packed_half_6 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_100),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_32 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_accumulator_26 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(0u));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xc900ull,
                                        matrix_accumulator_100, matrix_accumulator_101,
                                        matrix_accumulator_102, matrix_accumulator_103,
                                        memory_ranges, error_counts);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_6),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(0u));

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(matrix_fragment_32));

    matrix_fragment_18 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_151),
                            select_half_operand<2, false, false>(matrix_accumulator_151));

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                      select_half_operand<2, false, false>(matrix_accumulator_26));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                      select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_fragment_32 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                       select_half_operand<2, false, false>(packed_half_6));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xcd00ull,
                                        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14,
                                        matrix_fragment_15, memory_ranges, error_counts);

    packed_half_6 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_147),
                                        select_half_operand<2, false, false>(matrix_fragment_147));

    matrix_fragment_18 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_149),
                                      select_half_operand<2, false, false>(matrix_accumulator_149),
                                      select_half_operand<2, false, false>(matrix_fragment_18));

    packed_half_6 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_145),
                                      select_half_operand<2, false, false>(matrix_fragment_145),
                                      select_half_operand<2, false, false>(packed_half_6));

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_19));

    matrix_accumulator_132 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                       select_half_operand<2, false, false>(packed_half_6));

    packed_half_6 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                                   select_half_operand<2, false, false>(matrix_fragment_17));

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xcf00ull,
                                        matrix_fragment_16, matrix_fragment_17, matrix_fragment_18,
                                        matrix_fragment_19, memory_ranges, error_counts);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_132), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_133 = result_word;
    }

    matrix_fragment_33 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_130),
                                       select_half_operand<2, false, false>(matrix_fragment_33));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(packed_half_6), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_141 = result_word;
    }

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_half(0u, packed_half_4);

    matrix_fragment_122 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_126 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_126),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_7 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_121 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_131 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_122),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_122 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_126),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_7 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_34),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_126 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_121),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_35 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_35),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_122 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_123),
                                        select_half_operand<2, false, false>(matrix_fragment_122));

    matrix_fragment_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_120),
                      select_half_operand<2, false, false>(matrix_accumulator_131));

    matrix_fragment_121 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                                        select_half_operand<2, false, false>(matrix_accumulator_7));

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    matrix_fragment_123 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                        select_half_operand<2, false, false>(matrix_fragment_126));

    matrix_fragment_126 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                      select_half_operand<2, false, false>(matrix_accumulator_129));

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_88, matrix_fragment_89, 0u, 0u);

    matrix_fragment_88 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_132),
                       select_half_operand<2, false, false>(matrix_accumulator_133));

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_90, matrix_fragment_91, 0u, 0u);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_88), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_90 = result_word;
    }

    matrix_fragment_89 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                       select_half_operand<2, false, false>(matrix_accumulator_141));

    matrix_fragment_91 = permute_signed(packed_half_140, packed_half_140, 0x5432u);

    matrix_fragment_91 = add_half_pairs(select_half_operand<2, false, false>(packed_half_140),
                                        select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_91 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91), 0x4100410u);

    multiply_accumulate_half(packed_half_6, matrix_accumulator_7, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_accumulator_104,
                             matrix_accumulator_105);

    matrix_fragment_90 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                                        select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_104 = permute_signed(matrix_fragment_89, matrix_fragment_89, 0x5432u);

    matrix_accumulator_105 = half_to_float_bits((matrix_fragment_91 & 65535u) * 65537u);

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                       select_half_operand<2, false, false>(matrix_accumulator_104));

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_91 >> 16) * 65537u);

    matrix_fragment_89 = rsqrt_bits(matrix_accumulator_105);

    matrix_fragment_91 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x4100410u);

    multiply_accumulate_half(packed_half_140, matrix_accumulator_141, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_106,
                             matrix_accumulator_107);

    matrix_accumulator_105 = permute_signed(matrix_fragment_90, matrix_fragment_90, 0x5432u);

    matrix_accumulator_106 = half_to_float_bits((matrix_fragment_91 >> 16) * 65537u);

    matrix_accumulator_104 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_fragment_90 = half_to_float_bits((matrix_fragment_91 & 65535u) * 65537u);

    matrix_accumulator_142 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_142),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_105 = permute_signed(packed_half_158, packed_half_158, 0x5432u);

    matrix_accumulator_106 = rsqrt_bits(matrix_accumulator_106);

    matrix_fragment_127 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_127),
                      select_half_operand<2, false, false>(matrix_accumulator_142));

    multiply_accumulate_half(matrix_accumulator_142, matrix_accumulator_143, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_30, matrix_fragment_31, matrix_accumulator_108,
                             matrix_accumulator_109);

    matrix_fragment_90 = rsqrt_bits(matrix_fragment_90);

    matrix_accumulator_108 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_158),
                       select_half_operand<2, false, false>(matrix_accumulator_105));

    matrix_accumulator_105 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_104), 0x4100410u);

    matrix_fragment_91 = half_to_float_bits((matrix_accumulator_105 & 65535u) * 65537u);

    matrix_accumulator_107 = half_to_float_bits((matrix_accumulator_105 >> 16) * 65537u);

    matrix_accumulator_105 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_155),
                                           select_half_operand<2, false, false>(0u));

    matrix_accumulator_108 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_108), 0x4100410u);

    matrix_fragment_91 = rsqrt_bits(matrix_fragment_91);

    matrix_accumulator_105 =
        pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                      select_half_operand<2, false, false>(matrix_accumulator_105));

    packed_half_154 = pack_half(matrix_accumulator_106, matrix_fragment_90);

    matrix_accumulator_109 = half_to_float_bits((matrix_accumulator_108 & 65535u) * 65537u);

    matrix_fragment_90 = half_to_float_bits((matrix_accumulator_108 >> 16) * 65537u);

    matrix_accumulator_107 = rsqrt_bits(matrix_accumulator_107);

    matrix_fragment_88 = rsqrt_bits(matrix_fragment_88);

    matrix_accumulator_109 = rsqrt_bits(matrix_accumulator_109);

    matrix_fragment_90 = rsqrt_bits(matrix_fragment_90);

    matrix_fragment_91 = pack_half(matrix_accumulator_107, matrix_fragment_91);

    matrix_accumulator_148 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_148),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_150 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_150),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_108 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_145),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_accumulator_149 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_149),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_147 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_147),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_91 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_151),
                            select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_89 = pack_half(matrix_fragment_88, matrix_fragment_89);

    matrix_accumulator_106 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_144),
                            select_half_operand<2, false, false>(packed_half_154));

    matrix_accumulator_107 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_148),
                      select_half_operand<2, false, false>(0u));

    packed_half_154 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_146),
                                          select_half_operand<2, false, false>(packed_half_154));

    matrix_fragment_90 = pack_half(matrix_fragment_90, matrix_accumulator_109);

    matrix_accumulator_157 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_157),
                      select_half_operand<2, false, false>(0u));

    packed_half_155 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_150),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_149 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_149),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_106 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_106),
                      select_half_operand<2, false, false>(matrix_accumulator_107));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_104 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_156),
                      select_half_operand<2, false, false>(matrix_accumulator_157));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                            select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_107 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_154),
                                           select_half_operand<2, false, false>(packed_half_155));

    matrix_accumulator_108 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_108),
                      select_half_operand<2, false, false>(matrix_accumulator_149));

    matrix_accumulator_109 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_147),
                      select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_136 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_135 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<0, false, false>(packed_half_4));

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_32, matrix_fragment_33, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_138 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_136),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_139 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_88 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(matrix_fragment_89));

    multiply_accumulate_half(matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_fragment_126, matrix_fragment_127, matrix_accumulator_112,
                             matrix_accumulator_113);

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_104, matrix_accumulator_105, matrix_accumulator_114,
                             matrix_accumulator_115);

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_106, matrix_accumulator_107, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(packed_half_118, matrix_accumulator_119, matrix_fragment_120,
                             matrix_fragment_121, matrix_fragment_122, matrix_fragment_123,
                             matrix_accumulator_108, matrix_accumulator_109, packed_half_118,
                             matrix_accumulator_119);

    matrix_fragment_120 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_135),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_122 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                        select_half_operand<2, false, false>(0u));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_138),
                                       select_half_operand<2, false, false>(matrix_fragment_91));

    matrix_fragment_89 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(matrix_fragment_120));

    matrix_fragment_91 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_139),
                                       select_half_operand<2, false, false>(matrix_fragment_122));

    multiply_accumulate_half(matrix_fragment_122, matrix_fragment_123, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = 0x29c0u;

    packed_half_4 = permute_signed(packed_half_4, matrix_fragment_8, 0x5410u);

    packed_half_6 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_6),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_accumulator_100,
                             matrix_accumulator_101);

    matrix_accumulator_7 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_7),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_141 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_141),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    packed_half_6 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0x3c203c20u);

    matrix_accumulator_143 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_143),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_7 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7), 0x3c203c20u);

    matrix_accumulator_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_143 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_143), 0x3c203c20u);

    matrix_accumulator_113 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_113),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_120, matrix_fragment_121, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_102,
                             matrix_accumulator_103);

    matrix_accumulator_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_111 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3c203c20u);

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_110 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3c203c20u);

    matrix_accumulator_143 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_143), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_134, matrix_accumulator_135, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_accumulator_113 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_113), 0x3c203c20u);

    matrix_accumulator_114 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3c203c20u);

    matrix_accumulator_115 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3c203c20u);

    matrix_fragment_11 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7), 0x3e473e47u);

    matrix_accumulator_7 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_100, matrix_accumulator_101, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_fragment_126, matrix_fragment_127, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0x3e473e47u);

    packed_half_6 = maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_141),
                                       0x3c203c20u);

    packed_half_6 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_6), 0x3e473e47u);

    matrix_fragment_13 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3e473e47u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_accumulator_113 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_113), 0x3e473e47u);

    packed_half_140 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_140),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_114 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3e473e47u);

    matrix_accumulator_142 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_142),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_115 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3e473e47u);

    matrix_fragment_11 = (matrix_fragment_11 << 0x5u) + 0x7ff88000u;

    matrix_fragment_14 = (matrix_accumulator_143 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_112),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_8 = (matrix_accumulator_114 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_9 = (matrix_accumulator_115 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_140), 0x3c203c20u);

    matrix_fragment_17 = (packed_half_6 << 0x5u) + 0x7ff88000u;

    matrix_fragment_16 = (matrix_accumulator_7 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_7 = (matrix_accumulator_113 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                       select_half_operand<2, false, false>(matrix_fragment_17));

    matrix_accumulator_142 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_142), 0x3c203c20u);

    matrix_accumulator_115 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                       select_half_operand<2, false, false>(matrix_fragment_16));

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3e473e47u);

    matrix_accumulator_136 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_142 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_142), 0x3e473e47u);

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_112 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_112), 0x3c203c20u);

    matrix_accumulator_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    packed_half_118 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(packed_half_118),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_112 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_112), 0x3e473e47u);

    matrix_accumulator_136 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_114),
                       select_half_operand<2, false, false>(matrix_accumulator_136));

    matrix_fragment_12 = (matrix_fragment_12 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_15 = (matrix_fragment_15 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_88,
                             matrix_fragment_89, matrix_fragment_90, matrix_fragment_91,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_18,
                             matrix_fragment_19);

    matrix_fragment_10 = (matrix_accumulator_142 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_13 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_fragment_121 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_121),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_116 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3c203c20u);

    matrix_fragment_123 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_123),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    packed_half_118 =
        maximum_half_pairs(select_half_operand<2, false, false>(packed_half_118), 0x3c203c20u);

    matrix_accumulator_135 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    packed_half_6 = (matrix_accumulator_112 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                       select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_116 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3e473e47u);

    matrix_accumulator_113 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                       select_half_operand<2, false, false>(matrix_fragment_13));

    packed_half_118 =
        minimum_half_pairs(select_half_operand<2, false, false>(packed_half_118), 0x3e473e47u);

    matrix_accumulator_20 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3c203c20u);

    matrix_accumulator_21 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3c203c20u);

    matrix_fragment_121 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121), 0x3c203c20u);

    matrix_fragment_123 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123), 0x3c203c20u);

    matrix_accumulator_135 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3c203c20u);

    matrix_accumulator_112 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_accumulator_20 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3e473e47u);

    matrix_accumulator_137 =
        add_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_21 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3e473e47u);

    matrix_fragment_121 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121), 0x3e473e47u);

    matrix_fragment_123 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123), 0x3e473e47u);

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_135), 0x3e473e47u);

    matrix_fragment_19 = (matrix_accumulator_116 << 0x5u) + 0x7ff88000u;

    matrix_fragment_18 = (packed_half_118 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_137 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                       select_half_operand<2, false, false>(matrix_accumulator_137));

    matrix_fragment_88 = (matrix_accumulator_20 << 0x5u) + 0x7ff88000u;

    matrix_fragment_120 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_120),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_89 = (matrix_accumulator_21 << 0x5u) + 0x7ff88000u;

    matrix_fragment_122 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_122),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_112 = (matrix_fragment_121 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_134 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_20 = (matrix_fragment_123 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                       select_half_operand<2, false, false>(matrix_fragment_18));

    matrix_fragment_90 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_139 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                       select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_120 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120), 0x3c203c20u);

    matrix_accumulator_137 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_137),
                       select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_122 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122), 0x3c203c20u);

    matrix_accumulator_116 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_accumulator_134 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3c203c20u);

    matrix_accumulator_100 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_120 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120), 0x3e473e47u);

    matrix_accumulator_139 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                       select_half_operand<2, false, false>(matrix_accumulator_116));

    matrix_fragment_122 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122), 0x3e473e47u);

    matrix_accumulator_119 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_134), 0x3e473e47u);

    matrix_accumulator_116 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_102),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3c203c20u);

    matrix_accumulator_101 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_103 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_21 = (matrix_fragment_120 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_117 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_91 = (matrix_fragment_122 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_110 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_113 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_114 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_102 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_accumulator_111 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_119 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3c203c20u);

    matrix_accumulator_115 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_100 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_116), 0x3c203c20u);

    matrix_accumulator_101 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3c203c20u);

    matrix_accumulator_103 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3c203c20u);

    matrix_accumulator_138 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_accumulator_117 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3c203c20u);

    packed_half_118 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                                     select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_accumulator_135 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_119), 0x3e473e47u);

    matrix_accumulator_100 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_100), 0x3e473e47u);

    matrix_accumulator_110 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3c203c20u);

    matrix_accumulator_114 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3c203c20u);

    matrix_accumulator_101 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_101), 0x3e473e47u);

    matrix_accumulator_119 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_103), 0x3e473e47u);

    matrix_accumulator_111 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3c203c20u);

    matrix_accumulator_115 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3c203c20u);

    matrix_accumulator_138 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                       select_half_operand<2, false, false>(packed_half_118));

    matrix_accumulator_134 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_117), 0x3e473e47u);

    matrix_fragment_120 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_110), 0x3e473e47u);

    matrix_accumulator_114 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_114), 0x3e473e47u);

    matrix_fragment_121 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_111), 0x3e473e47u);

    matrix_accumulator_115 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_115), 0x3e473e47u);

    matrix_accumulator_102 = (matrix_accumulator_102 << 0x5u) + 0x7ff88000u;

    packed_half_118 = (matrix_accumulator_100 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_103 = (matrix_accumulator_101 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_119 = (matrix_accumulator_119 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_accumulator_134 = (matrix_accumulator_134 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_135 = (matrix_accumulator_135 << 0x5u) + 0x7ff88000u;

    matrix_fragment_120 = (matrix_fragment_120 << 0x5u) + 0x7ff88000u;

    matrix_fragment_122 = (matrix_accumulator_114 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                       select_half_operand<2, false, false>(matrix_accumulator_119));

    matrix_fragment_121 = (matrix_fragment_121 << 0x5u) + 0x7ff88000u;

    matrix_fragment_123 = (matrix_accumulator_115 << 0x5u) + 0x7ff88000u;

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_72, matrix_fragment_73, 0u, 0u);

    matrix_accumulator_100 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                       select_half_operand<2, false, false>(matrix_fragment_122));

    matrix_accumulator_101 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_139),
                       select_half_operand<2, false, false>(matrix_accumulator_101));

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                       select_half_operand<2, false, false>(packed_half_118));

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                       select_half_operand<2, false, false>(matrix_accumulator_135));

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_138),
                       select_half_operand<2, false, false>(matrix_fragment_73));

    matrix_accumulator_114 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                       select_half_operand<2, false, false>(matrix_fragment_123));

    matrix_fragment_72 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_136),
                       select_half_operand<2, false, false>(matrix_fragment_72));

    matrix_accumulator_136 = address_word_0 * 0x4u;

    matrix_fragment_73 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_73),
                       select_half_operand<2, false, false>(matrix_accumulator_100));

    matrix_accumulator_100 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_101),
                       select_half_operand<2, false, false>(matrix_accumulator_114));

    matrix_accumulator_101 = matrix_accumulator_136 & 0x1cu;

    matrix_accumulator_136 = condition_0 ? matrix_fragment_72 : matrix_accumulator_137;

    matrix_accumulator_139 = condition_0 ? matrix_accumulator_100 : matrix_fragment_73;

    matrix_fragment_72 = condition_0 ? matrix_accumulator_137 : matrix_fragment_72;

    matrix_fragment_73 = condition_0 ? matrix_fragment_73 : matrix_accumulator_100;

    matrix_accumulator_101 =
        uint32_t(int64_t(int32_t(address_word_0)) * (1ll << 0x1du) >> 32) + matrix_accumulator_101;

    matrix_accumulator_138 = !condition_2 ? matrix_accumulator_136 : matrix_accumulator_139;

    matrix_accumulator_136 = !condition_2 ? matrix_accumulator_139 : matrix_accumulator_136;

    matrix_accumulator_137 = !condition_2 ? matrix_fragment_72 : matrix_fragment_73;

    matrix_accumulator_139 = !condition_2 ? matrix_fragment_73 : matrix_fragment_72;

    matrix_accumulator_100 = 0x1u ^ matrix_accumulator_101;

    matrix_fragment_73 = 0x2u ^ matrix_accumulator_101;

    matrix_fragment_72 = 0x3u ^ matrix_accumulator_101;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_138), "r"(matrix_accumulator_101), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_138 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_137), "r"(matrix_accumulator_100), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_137 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_136), "r"(matrix_fragment_73), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_136 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_139), "r"(matrix_fragment_72), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_139 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_34, matrix_fragment_35, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_52, matrix_fragment_53, matrix_fragment_34,
                             matrix_fragment_35);

    packed_half_140 = !condition_3 ? matrix_accumulator_138 : matrix_accumulator_137;

    matrix_accumulator_141 = !condition_3 ? matrix_accumulator_137 : matrix_accumulator_138;

    multiply_accumulate_half(matrix_accumulator_130, matrix_accumulator_131, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_52, matrix_fragment_53, matrix_accumulator_130,
                             matrix_accumulator_131);

    matrix_accumulator_143 = !condition_3 ? matrix_accumulator_136 : matrix_accumulator_139;

    matrix_accumulator_142 = !condition_3 ? matrix_accumulator_139 : matrix_accumulator_136;

    matrix_fragment_52 = !condition_1 ? packed_half_140 : matrix_accumulator_143;

    matrix_fragment_53 = !condition_1 ? matrix_accumulator_141 : matrix_accumulator_142;

    packed_half_140 = !condition_1 ? matrix_accumulator_143 : packed_half_140;

    matrix_accumulator_141 = !condition_1 ? matrix_accumulator_142 : matrix_accumulator_141;

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_52 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                                        select_half_operand<2, false, false>(packed_half_140));

    matrix_fragment_52 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52),
                       select_half_operand<2, false, false>(matrix_accumulator_141));

    matrix_fragment_52 = add_half_pairs(select_half_operand<0, false, false>(matrix_fragment_52),
                                        select_half_operand<1, false, false>(matrix_fragment_52));

    matrix_fragment_52 = permute_signed(matrix_fragment_52, matrix_fragment_52, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_128, matrix_accumulator_129, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_128,
                             matrix_accumulator_129);

    matrix_fragment_52 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_52), 0x4100410u);

    matrix_accumulator_139 = uint32_t(int32_t(matrix_accumulator_2) >> 0x2u);

    multiply_accumulate_half(matrix_accumulator_132, matrix_accumulator_133, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_fragment_54, matrix_fragment_55, matrix_accumulator_132,
                             matrix_accumulator_133);

    matrix_fragment_54 = half_to_float_bits((matrix_fragment_52 & 65535u) * 65537u);

    matrix_fragment_55 = half_to_float_bits((matrix_fragment_52 >> 16) * 65537u);

    matrix_fragment_52 = matrix_accumulator_139 + 0x8u;

    matrix_fragment_54 = reciprocal_bits(matrix_fragment_54);

    matrix_fragment_53 = uint32_t(int32_t(matrix_fragment_52) >> 0x1fu);

    matrix_fragment_55 = reciprocal_bits(matrix_fragment_55);

    matrix_accumulator_138 = uint32_t(int32_t(matrix_accumulator_2) >> 0x1fu);

    matrix_accumulator_136 =
        uint32_t(combine_words(matrix_fragment_53, 0u) << 0x5u >> 32) + matrix_fragment_52;

    multiply_accumulate_half(matrix_accumulator_116, matrix_accumulator_117, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_116,
                             matrix_accumulator_117);

    multiply_accumulate_half(matrix_accumulator_110, matrix_accumulator_111, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_56, matrix_accumulator_57, matrix_accumulator_110,
                             matrix_accumulator_111);

    matrix_accumulator_137 = pack_half(matrix_fragment_55, matrix_fragment_54);

    matrix_accumulator_56 =
        uint32_t(combine_words(matrix_accumulator_138, 0u) << 0x5u >> 32) + matrix_accumulator_139;

    matrix_accumulator_57 = matrix_accumulator_136 & 0xffffffe0u;

    matrix_accumulator_56 = matrix_accumulator_56 & 0xffffffe0u;

    matrix_fragment_55 = transpose_half_tile(matrix_fragment_98);

    matrix_accumulator_136 = matrix_fragment_52 + (0u - matrix_accumulator_57);

    matrix_fragment_54 = transpose_half_tile(matrix_fragment_37);

    matrix_fragment_98 = matrix_accumulator_139 + (0u - matrix_accumulator_56);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_137), "r"(matrix_accumulator_136), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_136 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_137), "r"(matrix_fragment_98), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_57 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_accumulator_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                          select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_accumulator_74, matrix_accumulator_75, 0u, 0u);

    matrix_fragment_37 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_54 = transpose_half_tile(matrix_accumulator_124);

    matrix_fragment_99 = permute_signed(matrix_accumulator_136, matrix_accumulator_136, 0x5410u);

    matrix_fragment_16 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_16),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_124 = permute_signed(matrix_accumulator_57, matrix_accumulator_57, 0x5410u);

    matrix_fragment_17 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_17),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_57 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_56 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_accumulator_57));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_57 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_accumulator_139 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_114,
                             matrix_accumulator_115);

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_125),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_16 = uint32_t(int32_t(matrix_accumulator_57) >> 0x1fu);

    matrix_fragment_36 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_56));

    matrix_fragment_17 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_17));

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_58, matrix_accumulator_59, matrix_accumulator_74,
                             matrix_accumulator_75);

    matrix_accumulator_56 =
        uint32_t(combine_words(matrix_fragment_16, 0u) << 0x5u >> 32) + matrix_accumulator_57;

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_58 = uint32_t(int32_t(matrix_accumulator_139) >> 0x1fu);

    matrix_fragment_54 =
        uint32_t(combine_words(matrix_accumulator_58, 0u) << 0x5u >> 32) + matrix_accumulator_139;

    matrix_accumulator_93 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_56 = matrix_accumulator_56 & 0xffffffe0u;

    matrix_fragment_54 = matrix_fragment_54 & 0xffffffe0u;

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_55));

    matrix_fragment_10 = transpose_half_tile(matrix_fragment_39);

    matrix_fragment_16 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                       select_half_operand<2, false, false>(matrix_accumulator_93));

    matrix_accumulator_92 = matrix_accumulator_57 + (0u - matrix_accumulator_56);

    matrix_accumulator_136 = matrix_accumulator_139 + (0u - matrix_fragment_54);

    matrix_fragment_55 = transpose_half_tile(matrix_fragment_38);

    matrix_fragment_11 = transpose_half_tile(matrix_accumulator_85);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_137), "r"(matrix_accumulator_92), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_92 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_137), "r"(matrix_accumulator_136), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_136 = result_word;
    }

    matrix_accumulator_56 = transpose_half_tile(matrix_fragment_96);

    matrix_accumulator_57 = transpose_half_tile(matrix_accumulator_84);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_129 = transpose_half_tile(matrix_accumulator_129);

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_95 = permute_signed(matrix_accumulator_92, matrix_accumulator_92, 0x5410u);

    matrix_accumulator_133 = transpose_half_tile(matrix_accumulator_133);

    matrix_accumulator_94 = permute_signed(matrix_accumulator_136, matrix_accumulator_136, 0x5410u);

    matrix_accumulator_56 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                      select_half_operand<2, false, false>(matrix_fragment_97));

    matrix_accumulator_128 = transpose_half_tile(matrix_accumulator_128);

    matrix_accumulator_57 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_accumulator_132 = transpose_half_tile(matrix_accumulator_132);

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_accumulator_113 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_113),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_accumulator_112 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_112),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_90 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_90),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_113 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_113),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_112 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_112),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_90 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_accumulator_131 = transpose_half_tile(matrix_accumulator_131);

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_19),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_7),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_19 = transpose_half_tile(matrix_accumulator_130);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_accumulator_134 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_134),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_6),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_56, matrix_accumulator_57, 0u, 0u);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_88),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_91),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_89),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_14 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                      select_half_operand<2, false, false>(matrix_accumulator_113));

    matrix_fragment_13 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                      select_half_operand<2, false, false>(matrix_accumulator_112));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_90));

    matrix_fragment_99 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_135),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_accumulator_124 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_18),
                            select_half_operand<2, false, false>(matrix_accumulator_124));

    matrix_accumulator_112 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                           select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_accumulator_129 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_129),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_133),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_99 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_99),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_97 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_124),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                      select_half_operand<2, false, false>(matrix_accumulator_112));

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_128),
                                  select_half_operand<2, false, false>(matrix_accumulator_129));

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_132),
                      select_half_operand<2, false, false>(matrix_accumulator_7));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_134),
                                       select_half_operand<2, false, false>(matrix_fragment_99));

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_35 = transpose_half_tile(matrix_fragment_35);

    matrix_fragment_18 = transpose_half_tile(matrix_fragment_34);

    matrix_accumulator_117 = transpose_half_tile(matrix_accumulator_117);

    matrix_accumulator_111 = transpose_half_tile(matrix_accumulator_111);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_12,
                             matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                             matrix_accumulator_56, matrix_accumulator_57, 0u, 0u);

    matrix_accumulator_115 = transpose_half_tile(matrix_accumulator_115);

    matrix_accumulator_75 = transpose_half_tile(matrix_accumulator_75);

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_110 = transpose_half_tile(matrix_accumulator_110);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                      select_half_operand<2, false, false>(matrix_fragment_15));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                       select_half_operand<2, false, false>(matrix_fragment_97));

    packed_half_118 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_118),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_99 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_86, matrix_fragment_87) + 0x11110ull, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_6, matrix_accumulator_7, matrix_fragment_38,
                             matrix_fragment_39);

    matrix_fragment_34 = transpose_half_tile(matrix_accumulator_116);

    matrix_fragment_38 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_131),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_39 = transpose_half_tile(matrix_accumulator_74);

    matrix_fragment_19 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_19),
                                       select_half_operand<2, false, false>(matrix_fragment_38));

    matrix_fragment_38 = transpose_half_tile(matrix_accumulator_114);

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_35),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_117 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_117),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_18 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_18),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_111),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_115 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_115),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_96 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_34 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_34),
                      select_half_operand<2, false, false>(matrix_accumulator_117));

    matrix_fragment_35 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_110),
                                       select_half_operand<2, false, false>(matrix_fragment_35));

    matrix_fragment_120 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_120),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_39 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                       select_half_operand<2, false, false>(matrix_fragment_96));

    matrix_accumulator_95 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_122),
                            select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_38 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                      select_half_operand<2, false, false>(matrix_accumulator_115));

    matrix_accumulator_119 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_119),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_123 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_123),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_accumulator_75 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_118),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_95 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_74 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_119),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_34, matrix_fragment_35, matrix_accumulator_84,
                             matrix_accumulator_85);

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_92,
                             matrix_accumulator_93);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_121),
                            select_half_operand<2, false, false>(matrix_accumulator_94));

    matrix_accumulator_94 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_123),
                                          select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                                      select_half_operand<2, false, false>(matrix_accumulator_75));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_120),
                                       select_half_operand<2, false, false>(matrix_accumulator_95));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_accumulator_74));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_94));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_6, matrix_accumulator_7, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_12,
                             matrix_fragment_13);

    multiply_accumulate_half(matrix_fragment_88, matrix_fragment_89, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_88,
                             matrix_fragment_89);

    multiply_accumulate_half(matrix_fragment_90, matrix_fragment_91, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_90,
                             matrix_fragment_91);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_13 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_84),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_15 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_85),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_84 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_88),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_85 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_89),
                                          select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_86 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_90),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_91),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3, matrix_fragment_12,
                                         matrix_fragment_13, matrix_fragment_14, matrix_fragment_15,
                                         shared_bytes, memory_ranges, error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x400u, matrix_fragment_12, matrix_fragment_13,
        matrix_fragment_14, matrix_fragment_15, shared_bytes, memory_ranges, error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x400u, matrix_accumulator_84, matrix_accumulator_85,
        matrix_fragment_86, matrix_fragment_87, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10110ull, matrix_accumulator_84,
        matrix_accumulator_85, matrix_fragment_86, matrix_fragment_87, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10310ull, matrix_fragment_88,
        matrix_fragment_89, matrix_fragment_90, matrix_fragment_91, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10910ull, matrix_accumulator_92,
        matrix_accumulator_93, matrix_accumulator_94, matrix_accumulator_95, memory_ranges,
        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10b10ull, matrix_accumulator_112,
        matrix_accumulator_113, matrix_accumulator_114, matrix_accumulator_115, memory_ranges,
        error_counts);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_44, matrix_fragment_45, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_46, matrix_fragment_47, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_48, matrix_fragment_49, 0u, 0u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_accumulator_64,
                             matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67,
                             matrix_fragment_50, matrix_fragment_51, 0u, 0u);

    multiply_accumulate_half(matrix_fragment_50, matrix_fragment_51, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_accumulator_20,
                             matrix_accumulator_21);

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_74,
                             matrix_accumulator_75);

    multiply_accumulate_half(matrix_accumulator_74, matrix_accumulator_75, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_44 = unpack_fp8_bits(matrix_fragment_8);

    matrix_fragment_45 = unpack_fp8_bits(matrix_fragment_9);

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_accumulator_58,
                             matrix_accumulator_59);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_8 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_9 >> 16);

    multiply_accumulate_half(matrix_accumulator_58, matrix_accumulator_59, matrix_fragment_76,
                             matrix_fragment_77, matrix_fragment_78, matrix_fragment_79,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_96,
                             matrix_fragment_97);

    multiply_accumulate_half(matrix_fragment_96, matrix_fragment_97, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_46,
                             matrix_fragment_47);

    multiply_accumulate_half(matrix_accumulator_102, matrix_accumulator_103, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_68, matrix_accumulator_69, matrix_accumulator_102,
                             matrix_accumulator_103);

    multiply_accumulate_half(matrix_fragment_80, matrix_fragment_81, matrix_fragment_80,
                             matrix_fragment_81, matrix_fragment_82, matrix_fragment_83,
                             matrix_accumulator_70, matrix_accumulator_71, matrix_accumulator_64,
                             matrix_accumulator_65);

    matrix_fragment_8 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_9 = multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_76 = unpack_fp8_bits(matrix_fragment_10);

    matrix_fragment_10 = unpack_fp8_bits(matrix_fragment_10 >> 16);

    matrix_accumulator_62 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                            select_half_operand<2, false, false>(packed_half_162));

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_44,
                             matrix_fragment_45);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x400u, matrix_fragment_44,
                                        matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                                        shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_11);

    matrix_fragment_11 = unpack_fp8_bits(matrix_fragment_11 >> 16);

    matrix_fragment_9 = unpack_fp8_bits(matrix_fragment_13);

    matrix_accumulator_63 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_8 = unpack_fp8_bits(matrix_fragment_12);

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_11),
                            select_half_operand<2, false, false>(packed_half_162));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_9),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                            select_half_operand<2, false, false>(matrix_fragment_99));

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_88, matrix_fragment_89, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_12 >> 16);

    matrix_fragment_13 = unpack_fp8_bits(matrix_fragment_13 >> 16);

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(packed_half_160));

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(packed_half_160));

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_10,
                             matrix_fragment_11);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x600u, matrix_fragment_40,
                                        matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_76 = unpack_fp8_bits(matrix_fragment_14);

    multiply_accumulate_half(matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_accumulator_84, matrix_accumulator_85, matrix_fragment_8,
                             matrix_fragment_9);

    load_four_shared_words<CheckBounds>(shared_tile, packed_half_5 + 0x200u, matrix_fragment_8,
                                        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_fragment_14 = unpack_fp8_bits(matrix_fragment_14 >> 16);

    matrix_fragment_76 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_14 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(packed_half_162));

    multiply_accumulate_half(matrix_fragment_86, matrix_fragment_87, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_86, matrix_fragment_87, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_fragment_12 = unpack_fp8_bits(matrix_fragment_15);

    matrix_fragment_15 = unpack_fp8_bits(matrix_fragment_15 >> 16);

    matrix_fragment_77 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(packed_half_152));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(packed_half_162));

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_88, matrix_fragment_89, matrix_fragment_76,
                             matrix_fragment_77);

    packed_half_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_20));

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_44,
                             matrix_fragment_45, matrix_fragment_46, matrix_fragment_47,
                             matrix_fragment_90, matrix_fragment_91, matrix_fragment_14,
                             matrix_fragment_15);

    packed_half_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(matrix_fragment_50),
                                      select_half_operand<2, false, false>(packed_half_5));

    matrix_fragment_44 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                            select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_44 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_48),
                                      select_half_operand<2, false, false>(matrix_fragment_44));

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_66,
                             matrix_fragment_67);

    matrix_accumulator_68 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_81));

    matrix_fragment_67 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(matrix_fragment_80));

    matrix_fragment_66 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_accumulator_103));

    multiply_accumulate_half(matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_94, matrix_accumulator_95, matrix_fragment_86,
                             matrix_fragment_87);

    matrix_fragment_67 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                                      select_half_operand<2, false, false>(matrix_fragment_96),
                                      select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_fragment_66 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                                      select_half_operand<2, false, false>(matrix_accumulator_75),
                                      select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                                        select_half_operand<2, false, false>(packed_half_5));

    packed_half_5 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_21));

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_64,
                             matrix_accumulator_65);

    packed_half_5 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(matrix_fragment_51),
                                      select_half_operand<2, false, false>(packed_half_5));

    matrix_accumulator_68 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                                      select_half_operand<2, false, false>(matrix_fragment_97),
                                      select_half_operand<2, false, false>(matrix_accumulator_68));

    matrix_accumulator_64 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                            select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_accumulator_65 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(matrix_accumulator_102));

    matrix_accumulator_64 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_fragment_49),
                                      select_half_operand<2, false, false>(matrix_accumulator_64));

    matrix_accumulator_65 =
        fused_multiply_add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(packed_half_5));

    matrix_accumulator_65 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67),
                       select_half_operand<2, false, false>(matrix_accumulator_65));

    matrix_accumulator_68 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                       select_half_operand<2, false, false>(matrix_fragment_66));

    multiply_accumulate_half(matrix_accumulator_92, matrix_accumulator_93, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_92, matrix_accumulator_93, matrix_accumulator_84,
                             matrix_accumulator_85);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_66 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_68), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_67 = result_word;
    }

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_fragment_76,
                             matrix_fragment_77);

    packed_half_158 = threadIdx.y;

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_40,
                             matrix_fragment_41, matrix_fragment_42, matrix_fragment_43,
                             matrix_accumulator_114, matrix_accumulator_115, matrix_fragment_46,
                             matrix_fragment_47);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_64), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_43 = result_word;
    }

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_66));

    matrix_fragment_40 = parameters[parameter_output_low_2];
    matrix_fragment_41 = parameters[parameter_output_high_3];

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_65), "r"(0x2u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_42 = result_word;
    }

    matrix_fragment_82 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_68),
                                        select_half_operand<2, false, false>(matrix_fragment_67));

    uniform_condition_0 = int32_t(uniform_index_word_8) >= int32_t(uniform_index_word_10);

    uniform_index_word_5 = (uniform_index_word_5 << 0x3u) + uniform_index_word_6;

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_87), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_88 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_82), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_85 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_112, matrix_accumulator_113, matrix_accumulator_62,
                             matrix_accumulator_63);

    matrix_fragment_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_83 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_65),
                                        select_half_operand<2, false, false>(matrix_fragment_42));

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_86), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_89 = result_word;
    }

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_114, matrix_accumulator_115, matrix_accumulator_60,
                             matrix_accumulator_61);

    {
        uint32_t result_word;
        asm("shfl.sync.bfly.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_83), "r"(0x1u), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_84 = result_word;
    }

    uniform_index_word_7 = uniform_index_word_8 * uniform_index_word_11 + uniform_index_word_9;

    uniform_condition_0 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x4u) && !uniform_condition_0;

    packed_half_5 = packed_half_158 * 0x80u;

    uniform_condition_3 = uniform_condition_0 & uniform_condition_1;

    matrix_fragment_66 = uniform_index_word_7;

    matrix_accumulator_92 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_92),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_43 = matrix_fragment_66 * 0x100u + packed_half_5;

    matrix_accumulator_93 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_93),
                      select_half_operand<2, false, false>(0u));

    split_address_words(matrix_fragment_40, matrix_fragment_41,
                        int64_t(int32_t(matrix_fragment_43)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_fragment_40, matrix_fragment_41));
    if (uniform_condition_3) {
        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                          select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_14),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_11 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_60),
                          select_half_operand<2, false, false>(0u));

        matrix_accumulator_63 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_63),
                          select_half_operand<2, false, false>(matrix_fragment_10));

        matrix_fragment_14 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                           select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                          select_half_operand<2, false, false>(matrix_fragment_9));

        split_address_words(matrix_fragment_12, matrix_fragment_13,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_40, matrix_fragment_41));

        matrix_fragment_10 =
            pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_62),
                          select_half_operand<2, false, false>(matrix_fragment_11));

        matrix_fragment_9 = matrix_fragment_14;

        matrix_fragment_11 = matrix_accumulator_63;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_12, matrix_fragment_13), matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_2;

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_94),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_95),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_44 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_45 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_46),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_45 = permute_signed(matrix_fragment_45, matrix_fragment_10, 0x5410u);

        split_address_words(matrix_fragment_40, matrix_fragment_41,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_fragment_40, matrix_fragment_41));

        matrix_accumulator_93 = permute_signed(matrix_accumulator_93, matrix_fragment_8, 0x5410u);

        matrix_fragment_10 = permute_signed(matrix_fragment_44, matrix_fragment_11, 0x5410u);

        matrix_fragment_8 = permute_signed(matrix_accumulator_92, matrix_fragment_9, 0x5410u);

        matrix_fragment_9 = matrix_accumulator_93;

        matrix_fragment_11 = matrix_fragment_45;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_fragment_40, matrix_fragment_41) + 0x400ull, matrix_fragment_8,
            matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    __syncthreads();

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xd700ull,
                                        matrix_fragment_8, matrix_fragment_9, matrix_fragment_10,
                                        matrix_fragment_11, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xd100ull,
                                        matrix_accumulator_68, matrix_accumulator_69,
                                        matrix_accumulator_70, matrix_accumulator_71, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xd300ull,
                                        matrix_fragment_40, matrix_fragment_41, matrix_fragment_42,
                                        matrix_fragment_43, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xd500ull,
                                        matrix_accumulator_60, matrix_accumulator_61,
                                        matrix_accumulator_62, matrix_accumulator_63, memory_ranges,
                                        error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(address_word_24, address_word_25) + 0xd900ull, matrix_accumulator_64,
        matrix_accumulator_65, matrix_fragment_66, matrix_fragment_67, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xdb00ull,
                                        matrix_fragment_44, matrix_fragment_45, matrix_fragment_46,
                                        matrix_fragment_47, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xdd00ull,
                                        matrix_fragment_12, matrix_fragment_13, matrix_fragment_14,
                                        matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(combine_words(address_word_24, address_word_25) + 0xdf00ull,
                                        matrix_fragment_76, matrix_fragment_77, matrix_fragment_78,
                                        matrix_fragment_79, memory_ranges, error_counts);

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(matrix_fragment_88));

    matrix_fragment_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                        select_half_operand<2, false, false>(matrix_fragment_89));

    address_word_24 = permute_signed(matrix_fragment_87, matrix_fragment_87, 0x5432u);

    address_word_25 = permute_signed(matrix_fragment_86, matrix_fragment_86, 0x5432u);

    matrix_fragment_87 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87),
                                        select_half_operand<2, false, false>(address_word_24));

    matrix_fragment_86 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86),
                                        select_half_operand<2, false, false>(address_word_25));

    matrix_fragment_87 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_87), 0x4100410u);

    matrix_fragment_86 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_86), 0x4100410u);

    matrix_fragment_88 = half_to_float_bits((matrix_fragment_87 & 65535u) * 65537u);

    matrix_fragment_87 = half_to_float_bits((matrix_fragment_87 >> 16) * 65537u);

    address_word_24 = half_to_float_bits((matrix_fragment_86 & 65535u) * 65537u);

    matrix_fragment_86 = half_to_float_bits((matrix_fragment_86 >> 16) * 65537u);

    matrix_fragment_88 = rsqrt_bits(matrix_fragment_88);

    matrix_fragment_87 = rsqrt_bits(matrix_fragment_87);

    address_word_24 = rsqrt_bits(address_word_24);

    address_word_25 = rsqrt_bits(matrix_fragment_86);

    matrix_fragment_89 = pack_half(matrix_fragment_87, matrix_fragment_88);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_accumulator_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    address_word_25 = pack_half(address_word_25, address_word_24);

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(address_word_25));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(address_word_25));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(address_word_25));

    matrix_accumulator_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                            select_half_operand<2, false, false>(address_word_25));

    matrix_accumulator_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_49 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                            select_half_operand<0, false, false>(packed_half_4));

    address_word_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_84));

    matrix_fragment_87 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_82 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                       select_half_operand<2, false, false>(matrix_accumulator_85));

    matrix_fragment_51 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                       select_half_operand<2, false, false>(0u));

    matrix_accumulator_20 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_49),
                                          select_half_operand<2, false, false>(0u));

    matrix_accumulator_21 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_49 = permute_signed(matrix_fragment_51, matrix_accumulator_20, 0x5410u);

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_89));

    matrix_fragment_51 = permute_signed(matrix_accumulator_21, matrix_accumulator_58, 0x5410u);

    matrix_accumulator_20 = permute_signed(matrix_fragment_83, matrix_fragment_83, 0x5432u);

    matrix_accumulator_21 = permute_signed(matrix_fragment_82, matrix_fragment_82, 0x5432u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_83 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83),
                       select_half_operand<2, false, false>(matrix_accumulator_20));

    matrix_fragment_82 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_50 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(0u));

    address_word_25 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_83 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_83), 0x4100410u);

    matrix_fragment_82 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_82), 0x4100410u);

    matrix_fragment_48 = permute_signed(matrix_fragment_50, address_word_25, 0x5410u);

    matrix_fragment_50 = permute_signed(address_word_24, matrix_fragment_87, 0x5410u);

    address_word_24 = half_to_float_bits((matrix_fragment_83 & 65535u) * 65537u);

    matrix_fragment_83 = half_to_float_bits((matrix_fragment_83 >> 16) * 65537u);

    address_word_25 = half_to_float_bits((matrix_fragment_82 & 65535u) * 65537u);

    matrix_fragment_82 = half_to_float_bits((matrix_fragment_82 >> 16) * 65537u);

    address_word_24 = rsqrt_bits(address_word_24);

    matrix_fragment_83 = rsqrt_bits(matrix_fragment_83);

    address_word_25 = rsqrt_bits(address_word_25);

    matrix_fragment_82 = rsqrt_bits(matrix_fragment_82);

    address_word_24 = pack_half(matrix_fragment_83, address_word_24);

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<2, false, false>(address_word_24));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<2, false, false>(address_word_24));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<2, false, false>(address_word_24));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<2, false, false>(address_word_24));

    matrix_accumulator_74 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_74),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_96 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_96),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_102 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_102),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_80 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_80),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_accumulator_102 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_102),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_20, matrix_accumulator_21, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_8,
                             matrix_fragment_9);

    matrix_fragment_8 = pack_half(matrix_fragment_82, address_word_25);

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_75 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_75),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_97 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_97),
                            select_half_operand<0, false, false>(packed_half_4));

    multiply_accumulate_half(matrix_accumulator_68, matrix_accumulator_69, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_accumulator_68,
                             matrix_accumulator_69);

    matrix_accumulator_103 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_103),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_81 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_81),
                            select_half_operand<0, false, false>(packed_half_4));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_74),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_96),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_75 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_75),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_103 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_103),
                      select_half_operand<2, false, false>(0u));

    address_word_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_81),
                                    select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_accumulator_70, matrix_accumulator_71, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_70,
                             matrix_accumulator_71);

    matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_fragment_9, 0x5410u);

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_42,
                             matrix_fragment_43);

    multiply_accumulate_half(matrix_accumulator_60, matrix_accumulator_61, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_126, matrix_fragment_127, matrix_accumulator_60,
                             matrix_accumulator_61);

    multiply_accumulate_half(matrix_accumulator_62, matrix_accumulator_63, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_104, matrix_accumulator_105, matrix_accumulator_62,
                             matrix_accumulator_63);

    multiply_accumulate_half(matrix_fragment_48, matrix_fragment_49, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_10,
                             matrix_fragment_11);

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_97),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_80),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_9 = permute_signed(matrix_accumulator_75, matrix_fragment_10, 0x5410u);

    matrix_fragment_10 = permute_signed(matrix_accumulator_102, matrix_fragment_11, 0x5410u);

    matrix_fragment_11 = permute_signed(matrix_accumulator_103, address_word_24, 0x5410u);

    multiply_accumulate_half(matrix_accumulator_64, matrix_accumulator_65, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_26, matrix_accumulator_27, matrix_accumulator_64,
                             matrix_accumulator_65);

    multiply_accumulate_half(matrix_fragment_66, matrix_fragment_67, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_28, matrix_fragment_29, matrix_fragment_66,
                             matrix_fragment_67);

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_44,
                             matrix_fragment_45);

    multiply_accumulate_half(matrix_fragment_46, matrix_fragment_47, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_32, matrix_fragment_33, matrix_fragment_46,
                             matrix_fragment_47);

    matrix_fragment_41 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_41),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_68 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_14, matrix_fragment_15, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_104, matrix_accumulator_105, matrix_fragment_14,
                             matrix_fragment_15);

    matrix_accumulator_70 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_40 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_40),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_42 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_42),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_69 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    multiply_accumulate_half(matrix_fragment_12, matrix_fragment_13, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_126, matrix_fragment_127, matrix_fragment_12,
                             matrix_fragment_13);

    matrix_accumulator_71 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_43 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_43),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_41 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3c203c20u);

    matrix_accumulator_64 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_68 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3c203c20u);

    matrix_fragment_66 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_66),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_70 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3c203c20u);

    matrix_fragment_44 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_44),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_40 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_76, matrix_fragment_77, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_106, matrix_accumulator_107, matrix_fragment_76,
                             matrix_fragment_77);

    matrix_fragment_42 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3c203c20u);

    matrix_fragment_46 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_46),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_69 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3c203c20u);

    matrix_accumulator_71 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3c203c20u);

    matrix_fragment_43 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3c203c20u);

    multiply_accumulate_half(matrix_fragment_78, matrix_fragment_79, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_108, matrix_accumulator_109, matrix_fragment_78,
                             matrix_fragment_79);

    matrix_fragment_41 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41), 0x3e473e47u);

    matrix_accumulator_68 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_68), 0x3e473e47u);

    matrix_accumulator_70 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_70), 0x3e473e47u);

    matrix_fragment_40 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40), 0x3e473e47u);

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42), 0x3e473e47u);

    matrix_accumulator_69 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_69), 0x3e473e47u);

    matrix_accumulator_71 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_71), 0x3e473e47u);

    matrix_fragment_43 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43), 0x3e473e47u);

    matrix_accumulator_64 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3c203c20u);

    matrix_fragment_66 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3c203c20u);

    matrix_fragment_44 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3c203c20u);

    matrix_fragment_46 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3c203c20u);

    matrix_accumulator_61 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_33 = (matrix_fragment_41 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_63 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_64 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_64), 0x3e473e47u);

    matrix_fragment_66 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_66), 0x3e473e47u);

    matrix_fragment_44 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44), 0x3e473e47u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46), 0x3e473e47u);

    matrix_fragment_28 = (matrix_accumulator_68 << 0x5u) + 0x7ff88000u;

    matrix_fragment_31 = (matrix_accumulator_70 << 0x5u) + 0x7ff88000u;

    matrix_fragment_32 = (matrix_fragment_40 << 0x5u) + 0x7ff88000u;

    matrix_fragment_41 = (matrix_fragment_42 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_60 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_29 = (matrix_accumulator_69 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_62 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_30 = (matrix_accumulator_71 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_65 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_40 = (matrix_fragment_43 << 0x5u) + 0x7ff88000u;

    matrix_fragment_67 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_67),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_61 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3c203c20u);

    matrix_fragment_45 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_45),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_63 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3c203c20u);

    matrix_fragment_47 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_47),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_14 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_14),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    address_word_24 = (matrix_accumulator_64 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                                        select_half_operand<2, false, false>(matrix_fragment_31));

    address_word_25 = (matrix_fragment_66 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                                        select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_26 = (matrix_fragment_44 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                                        select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_accumulator_27 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                                        select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_accumulator_60 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3c203c20u);

    matrix_fragment_12 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_12),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_61 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_61), 0x3e473e47u);

    matrix_fragment_13 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_13),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_62 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3c203c20u);

    matrix_fragment_15 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_15),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_65 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3c203c20u);

    matrix_accumulator_63 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_63), 0x3e473e47u);

    matrix_fragment_67 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3c203c20u);

    matrix_fragment_45 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3c203c20u);

    matrix_fragment_47 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3c203c20u);

    matrix_fragment_14 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3c203c20u);

    matrix_fragment_50 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_60 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_60), 0x3e473e47u);

    matrix_fragment_42 = add_half_pairs(select_half_operand<2, false, false>(address_word_24),
                                        select_half_operand<2, false, false>(address_word_25));

    matrix_accumulator_62 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_62), 0x3e473e47u);

    matrix_fragment_43 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                       select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_accumulator_65 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_65), 0x3e473e47u);

    matrix_accumulator_58 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                       select_half_operand<2, false, false>(matrix_fragment_46));

    matrix_fragment_67 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_67), 0x3e473e47u);

    matrix_accumulator_20 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_12 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3c203c20u);

    matrix_accumulator_21 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_46 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45), 0x3e473e47u);

    matrix_fragment_48 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_48),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_47 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47), 0x3e473e47u);

    matrix_fragment_8 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14), 0x3e473e47u);

    matrix_fragment_49 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_49),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_14 = (matrix_accumulator_61 << 0x5u) + 0x7ff88000u;

    matrix_fragment_76 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_76),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_45 = (matrix_accumulator_63 << 0x5u) + 0x7ff88000u;

    matrix_fragment_77 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_77),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_13 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3c203c20u);

    matrix_fragment_78 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_78),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_fragment_15 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3c203c20u);

    matrix_fragment_79 = fused_multiply_add_half_pairs(
        select_half_operand<2, false, false>(matrix_fragment_79),
        select_half_operand<1, false, false>(packed_half_4), 0x3d343d34u);

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_42 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12), 0x3e473e47u);

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                                       select_half_operand<2, false, false>(matrix_fragment_45));

    matrix_fragment_12 = (matrix_accumulator_60 << 0x5u) + 0x7ff88000u;

    matrix_fragment_43 = (matrix_accumulator_62 << 0x5u) + 0x7ff88000u;

    matrix_fragment_44 = (matrix_accumulator_65 << 0x5u) + 0x7ff88000u;

    matrix_fragment_51 = (matrix_fragment_67 << 0x5u) + 0x7ff88000u;

    matrix_fragment_46 = (matrix_fragment_46 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_59 = (matrix_fragment_47 << 0x5u) + 0x7ff88000u;

    matrix_fragment_13 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13), 0x3e473e47u);

    matrix_accumulator_20 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3c203c20u);

    matrix_accumulator_21 = maximum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3c203c20u);

    matrix_fragment_48 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3c203c20u);

    matrix_fragment_15 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15), 0x3e473e47u);

    matrix_fragment_49 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3c203c20u);

    matrix_fragment_76 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3c203c20u);

    matrix_fragment_77 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3c203c20u);

    matrix_fragment_78 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3c203c20u);

    matrix_fragment_79 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3c203c20u);

    matrix_accumulator_60 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                       select_half_operand<2, false, false>(matrix_fragment_51));

    matrix_fragment_47 = (matrix_fragment_8 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                                       select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_accumulator_20 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_20), 0x3e473e47u);

    matrix_accumulator_61 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                       select_half_operand<2, false, false>(matrix_accumulator_59));

    matrix_accumulator_21 = minimum_half_pairs(
        select_half_operand<2, false, false>(matrix_accumulator_21), 0x3e473e47u);

    matrix_fragment_10 = add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                                        select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_48 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48), 0x3e473e47u);

    packed_half_4 = (matrix_fragment_13 << 0x5u) + 0x7ff88000u;

    matrix_fragment_42 = (matrix_fragment_42 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_58 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49), 0x3e473e47u);

    matrix_fragment_76 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_76), 0x3e473e47u);

    matrix_fragment_77 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_77), 0x3e473e47u);

    matrix_fragment_78 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_78), 0x3e473e47u);

    matrix_fragment_79 =
        minimum_half_pairs(select_half_operand<2, false, false>(matrix_fragment_79), 0x3e473e47u);

    matrix_fragment_13 = (matrix_fragment_15 << 0x5u) + 0x7ff88000u;

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_fragment_50 = (matrix_accumulator_20 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_60 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_49 = (matrix_accumulator_21 << 0x5u) + 0x7ff88000u;

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                                        select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_accumulator_61 = (matrix_fragment_48 << 0x5u) + 0x7ff88000u;

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                                       select_half_operand<2, false, false>(matrix_fragment_13));

    matrix_accumulator_58 = (matrix_accumulator_58 << 0x5u) + 0x7ff88000u;

    matrix_fragment_15 = (matrix_fragment_76 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_20 = (matrix_fragment_77 << 0x5u) + 0x7ff88000u;

    matrix_fragment_48 = (matrix_fragment_78 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_21 = (matrix_fragment_79 << 0x5u) + 0x7ff88000u;

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_62 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_60),
                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_11 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                       select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_accumulator_60 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                       select_half_operand<2, false, false>(matrix_fragment_48));

    matrix_accumulator_63 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                       select_half_operand<2, false, false>(matrix_accumulator_21));

    matrix_fragment_8 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_8),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_11 = add_half_pairs(select_half_operand<2, false, false>(matrix_fragment_10),
                                        select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_accumulator_60 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_63 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_62),
                       select_half_operand<2, false, false>(matrix_accumulator_63));

    matrix_fragment_9 = condition_0 ? matrix_fragment_11 : matrix_fragment_8;

    matrix_fragment_8 = condition_0 ? matrix_fragment_8 : matrix_fragment_11;

    matrix_fragment_10 = condition_0 ? matrix_accumulator_63 : matrix_accumulator_60;

    matrix_accumulator_63 = condition_0 ? matrix_accumulator_60 : matrix_accumulator_63;

    matrix_accumulator_60 = !condition_2 ? matrix_fragment_9 : matrix_fragment_10;

    matrix_fragment_11 = !condition_2 ? matrix_fragment_8 : matrix_accumulator_63;

    matrix_fragment_10 = !condition_2 ? matrix_fragment_10 : matrix_fragment_9;

    matrix_accumulator_63 = !condition_2 ? matrix_accumulator_63 : matrix_fragment_8;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_60), "r"(matrix_accumulator_101), "r"(0x1fu),
              "r"(__activemask()));
        matrix_accumulator_60 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_11), "r"(matrix_accumulator_100), "r"(0x1fu),
              "r"(__activemask()));
        matrix_fragment_11 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_10), "r"(matrix_fragment_73), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_10 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_accumulator_63), "r"(matrix_fragment_72), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_63 = result_word;
    }

    matrix_fragment_8 = !condition_3 ? matrix_accumulator_60 : matrix_fragment_11;

    matrix_fragment_9 = !condition_3 ? matrix_fragment_11 : matrix_accumulator_60;

    matrix_accumulator_65 = !condition_3 ? matrix_fragment_10 : matrix_accumulator_63;

    matrix_accumulator_62 = !condition_3 ? matrix_accumulator_63 : matrix_fragment_10;

    matrix_accumulator_64 = !condition_1 ? matrix_fragment_8 : matrix_accumulator_65;

    matrix_fragment_67 = !condition_1 ? matrix_fragment_9 : matrix_accumulator_62;

    matrix_fragment_8 = !condition_1 ? matrix_accumulator_65 : matrix_fragment_8;

    matrix_fragment_9 = !condition_1 ? matrix_accumulator_62 : matrix_fragment_9;

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_fragment_67));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_fragment_8));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64),
                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_accumulator_64 =
        add_half_pairs(select_half_operand<0, false, false>(matrix_accumulator_64),
                       select_half_operand<1, false, false>(matrix_accumulator_64));

    matrix_accumulator_64 = permute_signed(matrix_accumulator_64, matrix_accumulator_64, 0x5410u);

    matrix_accumulator_64 =
        maximum_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_64), 0x4100410u);

    matrix_fragment_8 = half_to_float_bits((matrix_accumulator_64 & 65535u) * 65537u);

    matrix_accumulator_64 = half_to_float_bits((matrix_accumulator_64 >> 16) * 65537u);

    matrix_fragment_8 = reciprocal_bits(matrix_fragment_8);

    matrix_fragment_53 =
        uint32_t(combine_words(matrix_fragment_53, 0u) << 0x5u >> 32) + matrix_fragment_52;

    matrix_fragment_9 = reciprocal_bits(matrix_accumulator_64);

    matrix_fragment_53 = matrix_fragment_53 & 0xffffffe0u;

    matrix_fragment_11 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x10u;

    matrix_fragment_10 = matrix_fragment_52 + (0u - matrix_fragment_53);

    matrix_fragment_52 =
        uint32_t(int64_t(int32_t(matrix_accumulator_2)) * (1ll << 0x1eu) >> 32) + 0x18u;

    matrix_fragment_53 = uint32_t(int32_t(matrix_fragment_52) >> 0x1fu);

    matrix_fragment_9 = pack_half(matrix_fragment_9, matrix_fragment_8);

    matrix_fragment_8 = uint32_t(int32_t(matrix_fragment_11) >> 0x1fu);

    matrix_fragment_53 =
        uint32_t(combine_words(matrix_fragment_53, 0u) << 0x5u >> 32) + matrix_fragment_52;

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_fragment_98), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_98 = result_word;
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

    matrix_fragment_53 = matrix_fragment_53 & 0xffffffe0u;

    matrix_fragment_8 = matrix_fragment_8 & 0xffffffe0u;

    matrix_accumulator_60 = matrix_fragment_52 + (0u - matrix_fragment_53);

    matrix_accumulator_62 = matrix_fragment_11 + (0u - matrix_fragment_8);

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_60), "r"(0x1fu), "r"(__activemask()));
        matrix_accumulator_60 = result_word;
    }

    {
        uint32_t result_word;
        asm("shfl.sync.idx.b32 %0,%1,%2,%3,%4;"
            : "=r"(result_word)
            : "r"(matrix_fragment_9), "r"(matrix_accumulator_62), "r"(0x1fu), "r"(__activemask()));
        matrix_fragment_53 = result_word;
    }

    matrix_fragment_98 = permute_signed(matrix_fragment_98, matrix_fragment_98, 0x5410u);

    matrix_fragment_52 = permute_signed(matrix_fragment_10, matrix_fragment_10, 0x5410u);

    matrix_fragment_31 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_31),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_41 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_41),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_30 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_30),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_40 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_40),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_28),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_29 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_29),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_41 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_32),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_40 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_33),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_53 = permute_signed(matrix_fragment_53, matrix_fragment_53, 0x5410u);

    matrix_accumulator_60 = permute_signed(matrix_accumulator_60, matrix_accumulator_60, 0x5410u);

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(matrix_fragment_31));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                      select_half_operand<2, false, false>(matrix_fragment_30));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_fragment_40));

    address_word_25 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_25),
                                          select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_51 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_51),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_59 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_59),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    address_word_25 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_25),
                                    select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_45 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_45),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_43 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_43),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_56, matrix_accumulator_57, 0u, 0u);

    matrix_fragment_8 =
        multiply_half_pairs(select_half_operand<2, false, false>(address_word_24),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    address_word_24 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_51),
                                    select_half_operand<2, false, false>(0u));

    matrix_fragment_10 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_26 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_59),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_44),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_46),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_8),
                                      select_half_operand<2, false, false>(address_word_25));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(address_word_24));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_10),
                                       select_half_operand<2, false, false>(matrix_accumulator_27));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_26));

    matrix_accumulator_61 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_61),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_58 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_58),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_12 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_12),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_fragment_90 = threadIdx.y;

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_36, matrix_fragment_37, 0u, 0u);

    matrix_fragment_50 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_50),
                            select_half_operand<2, false, false>(matrix_fragment_98));

    matrix_accumulator_61 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_61),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_47 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_47),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_accumulator_58 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_58),
                      select_half_operand<2, false, false>(0u));

    matrix_fragment_48 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_48),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0xc00u, address_word_24, address_word_25,
        matrix_accumulator_26, matrix_accumulator_27, shared_bytes, memory_ranges, error_counts);

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_16, matrix_fragment_17, 0u, 0u);

    matrix_fragment_13 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_13),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_21 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_21),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_54, matrix_fragment_55, 0u, 0u);

    matrix_fragment_42 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_42),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_accumulator_56, matrix_accumulator_57, 0u, 0u);

    matrix_fragment_9 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_14),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_49),
                            select_half_operand<2, false, false>(matrix_fragment_52));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_12),
                                      select_half_operand<2, false, false>(matrix_fragment_43));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_9),
                                      select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_50),
                                       select_half_operand<2, false, false>(matrix_accumulator_61));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_11),
                                       select_half_operand<2, false, false>(matrix_accumulator_58));

    matrix_fragment_15 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_15),
                            select_half_operand<2, false, false>(matrix_fragment_53));

    matrix_fragment_47 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_47),
                                       select_half_operand<2, false, false>(0u));

    packed_half_4 =
        multiply_half_pairs(select_half_operand<2, false, false>(packed_half_4),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    matrix_accumulator_20 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_20),
                            select_half_operand<2, false, false>(matrix_accumulator_60));

    multiply_accumulate_half(matrix_fragment_40, matrix_fragment_41, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_40,
                             matrix_fragment_41);

    multiply_accumulate_half(matrix_fragment_32, matrix_fragment_33, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_6, matrix_accumulator_7, matrix_fragment_32,
                             matrix_fragment_33);

    multiply_accumulate_half(matrix_fragment_30, matrix_fragment_31, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_30,
                             matrix_fragment_31);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_fragment_28,
                             matrix_fragment_29);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_13),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_48),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_21),
                                       select_half_operand<2, false, false>(0u));

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                      select_half_operand<2, false, false>(matrix_fragment_47));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_4),
                                      select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_15),
                                       select_half_operand<2, false, false>(matrix_fragment_10));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_20),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             packed_half_6, matrix_accumulator_7, matrix_fragment_36,
                             matrix_fragment_37);

    matrix_accumulator_7 = matrix_accumulator_2 & 0x7ffffffcu;

    multiply_accumulate_half(matrix_accumulator_56, matrix_accumulator_57, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_38, matrix_fragment_39, matrix_accumulator_56,
                             matrix_accumulator_57);

    packed_half_4 = 0u - matrix_accumulator_7 + address_word_0;

    packed_half_6 = parameters[parameter_trained_weights_low_4];
    matrix_accumulator_7 = parameters[parameter_trained_weights_high_5];

    multiply_accumulate_half(matrix_fragment_16, matrix_fragment_17, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_16,
                             matrix_fragment_17);

    matrix_fragment_90 = matrix_fragment_90 * 0x20u;

    multiply_accumulate_half(matrix_fragment_54, matrix_fragment_55, matrix_fragment_8,
                             matrix_fragment_9, matrix_fragment_10, matrix_fragment_11,
                             matrix_fragment_34, matrix_fragment_35, matrix_fragment_54,
                             matrix_fragment_55);

    matrix_fragment_9 = matrix_accumulator_2 & 0xfffffffcu;

    matrix_fragment_8 = packed_half_4 * 0x2u + matrix_fragment_90;

    matrix_accumulator_21 = reused_word_165 + address_word_0 + (0u - matrix_fragment_9);

    matrix_fragment_15 = reused_word_164 + address_word_0 + (0u - matrix_fragment_9);

    index_word_159 = index_word_159 + address_word_0 + (0u - matrix_fragment_9);

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_32),
                                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_33),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                  select_half_operand<2, false, false>(0u));

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_40),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_13 = uint32_t(int32_t(matrix_fragment_8) >> 0x1u);

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_41),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    matrix_fragment_30 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_30),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_31 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_31),
                                       select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_2 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                         select_half_operand<2, false, false>(0u));

    matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_56),
                                       select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_57),
                                  select_half_operand<2, false, false>(0u));

    split_address_words(matrix_accumulator_20, matrix_accumulator_21,
                        uint64_t(uint32_t(matrix_accumulator_21)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_6, matrix_accumulator_7));

    matrix_fragment_52 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_16),
                                       select_half_operand<2, false, false>(matrix_fragment_9));

    split_address_words(matrix_fragment_32, matrix_fragment_33,
                        uint64_t(uint32_t(matrix_fragment_15)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_6, matrix_accumulator_7));

    matrix_fragment_53 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_17),
                                       select_half_operand<2, false, false>(matrix_accumulator_2));

    store_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_3, matrix_fragment_28,
                                         matrix_fragment_29, matrix_fragment_30, matrix_fragment_31,
                                         shared_bytes, memory_ranges, error_counts);

    split_address_words(matrix_fragment_8, matrix_fragment_9,
                        int64_t(int32_t(matrix_fragment_13)) * int64_t(int32_t(0x4u)) +
                            combine_words(packed_half_6, matrix_accumulator_7));

    matrix_fragment_54 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_54),
                                       select_half_operand<2, false, false>(matrix_fragment_11));

    matrix_fragment_40 = load_aligned_word<CheckBounds>(
        combine_words(matrix_accumulator_20, matrix_accumulator_21) + 0x11110ull, memory_ranges,
        error_counts);

    matrix_fragment_55 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_55),
                                       select_half_operand<2, false, false>(packed_half_4));

    split_address_words(packed_half_6, matrix_accumulator_7,
                        uint64_t(uint32_t(index_word_159)) * uint64_t(uint32_t(0x4u)) +
                            combine_words(packed_half_6, matrix_accumulator_7));

    packed_half_4 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_8, matrix_fragment_9) + 0x11110ull, memory_ranges,
        error_counts);

    store_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x400u, matrix_fragment_52, matrix_fragment_53,
        matrix_fragment_54, matrix_fragment_55, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_41 = load_aligned_word<CheckBounds>(
        combine_words(matrix_fragment_32, matrix_fragment_33) + 0x11110ull, memory_ranges,
        error_counts);

    packed_half_6 = load_aligned_word<CheckBounds>(
        combine_words(packed_half_6, matrix_accumulator_7) + 0x11110ull, memory_ranges,
        error_counts);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_3 + 0x800u, matrix_fragment_36, matrix_fragment_37,
        matrix_fragment_38, matrix_fragment_39, shared_bytes, memory_ranges, error_counts);

    __syncthreads();

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10110ull, matrix_fragment_28,
        matrix_fragment_29, matrix_fragment_30, matrix_fragment_31, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10310ull, matrix_fragment_12,
        matrix_fragment_13, matrix_fragment_14, matrix_fragment_15, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10910ull, matrix_fragment_16,
        matrix_fragment_17, matrix_fragment_18, matrix_fragment_19, memory_ranges, error_counts);

    load_four_words_cached<CheckBounds>(
        combine_words(matrix_fragment_22, matrix_fragment_23) + 0x10b10ull, matrix_fragment_8,
        matrix_fragment_9, matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);

    uniform_index_word_7 = 0u;

    uniform_index_word_6 = 0x400u;

    uniform_index_word_6 = (uniform_index_word_7 << 0x18u) + uniform_index_word_6;

    matrix_accumulator_7 = (address_word_0 << 0x4u) + uniform_index_word_6;

    load_four_shared_words<CheckBounds>(shared_tile, matrix_accumulator_7, matrix_fragment_32,
                                        matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                                        shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_36);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_7 + 0x400u, matrix_accumulator_20, matrix_accumulator_21,
        matrix_fragment_22, matrix_fragment_23, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_37);

    matrix_fragment_36 = unpack_fp8_bits(matrix_fragment_36 >> 16);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_7 + 0x600u, matrix_fragment_48, matrix_fragment_49,
        matrix_fragment_50, matrix_fragment_51, shared_bytes, memory_ranges, error_counts);

    matrix_fragment_37 = unpack_fp8_bits(matrix_fragment_37 >> 16);

    uniform_index_word_11 = uniform_index_word_8 * uniform_index_word_11 + uniform_index_word_11;

    uniform_index_word_8 = uniform_index_word_8 + 0x1u;

    uniform_condition_0 = int32_t(uniform_index_word_8) >= int32_t(uniform_index_word_10);

    uniform_index_word_9 = uniform_index_word_9 + uniform_index_word_11;

    uniform_condition_0 =
        int32_t(uniform_index_word_5) > int32_t(0u - 0x8u) && !uniform_condition_0;

    uniform_condition_1 = uniform_condition_0 & uniform_condition_1;

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_36 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_36),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    matrix_fragment_37 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_37),
                            select_half_operand<2, false, false>(matrix_fragment_40));

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(matrix_fragment_38);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_fragment_39);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = unpack_fp8_bits(address_word_24);

    matrix_accumulator_3 = unpack_fp8_bits(address_word_25);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(packed_half_4));

    matrix_fragment_38 = unpack_fp8_bits(matrix_fragment_38 >> 16);

    matrix_fragment_39 = unpack_fp8_bits(matrix_fragment_39 >> 16);

    matrix_fragment_38 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_38),
                            select_half_operand<2, false, false>(packed_half_6));

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_28, matrix_fragment_29, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_fragment_39 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_fragment_39),
                            select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_2 = unpack_fp8_bits(matrix_accumulator_26);

    matrix_accumulator_3 = unpack_fp8_bits(matrix_accumulator_27);

    matrix_accumulator_2 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_2),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    matrix_accumulator_3 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_3),
                            select_half_operand<2, false, false>(matrix_fragment_41));

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_30, matrix_fragment_31, matrix_fragment_36,
                             matrix_fragment_37);

    address_word_24 = unpack_fp8_bits(address_word_24 >> 16);

    address_word_25 = unpack_fp8_bits(address_word_25 >> 16);

    matrix_accumulator_26 = unpack_fp8_bits(matrix_accumulator_26 >> 16);

    matrix_accumulator_27 = unpack_fp8_bits(matrix_accumulator_27 >> 16);

    address_word_24 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_24),
                                          select_half_operand<2, false, false>(matrix_fragment_40));

    address_word_25 = multiply_half_pairs(select_half_operand<2, false, false>(address_word_25),
                                          select_half_operand<2, false, false>(matrix_fragment_40));

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_14, matrix_fragment_15, matrix_fragment_38,
                             matrix_fragment_39);

    load_four_shared_words<CheckBounds>(
        shared_tile, matrix_accumulator_7 + 0x200u, matrix_fragment_32, matrix_fragment_33,
        matrix_fragment_34, matrix_fragment_35, shared_bytes, memory_ranges, error_counts);

    matrix_accumulator_26 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_26),
                            select_half_operand<2, false, false>(packed_half_6));

    matrix_accumulator_27 =
        multiply_half_pairs(select_half_operand<2, false, false>(matrix_accumulator_27),
                            select_half_operand<2, false, false>(packed_half_6));

    multiply_accumulate_half(matrix_accumulator_2, matrix_accumulator_3, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_12, matrix_fragment_13, matrix_accumulator_2,
                             matrix_accumulator_3);

    packed_half_4 = threadIdx.x;

    multiply_accumulate_half(address_word_24, address_word_25, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_30, matrix_fragment_31, address_word_24,
                             address_word_25);

    matrix_fragment_13 = threadIdx.z;

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_accumulator_20,
                             matrix_accumulator_21, matrix_fragment_22, matrix_fragment_23,
                             matrix_fragment_14, matrix_fragment_15, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_28, matrix_fragment_29, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_28,
                             matrix_fragment_29);

    multiply_accumulate_half(packed_half_6, matrix_accumulator_7, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_8, matrix_fragment_9, matrix_accumulator_2,
                             matrix_accumulator_3);

    matrix_accumulator_2 = parameters[parameter_output_low_2];
    matrix_accumulator_3 = parameters[parameter_output_high_3];

    multiply_accumulate_half(matrix_fragment_42, matrix_fragment_43, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_16, matrix_fragment_17, matrix_fragment_42,
                             matrix_fragment_43);

    multiply_accumulate_half(matrix_fragment_36, matrix_fragment_37, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_18, matrix_fragment_19, matrix_fragment_36,
                             matrix_fragment_37);

    multiply_accumulate_half(address_word_24, address_word_25, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_18, matrix_fragment_19, address_word_24,
                             address_word_25);

    multiply_accumulate_half(matrix_accumulator_26, matrix_accumulator_27, matrix_fragment_48,
                             matrix_fragment_49, matrix_fragment_50, matrix_fragment_51,
                             matrix_fragment_10, matrix_fragment_11, matrix_accumulator_26,
                             matrix_accumulator_27);

    multiply_accumulate_half(matrix_fragment_38, matrix_fragment_39, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_10, matrix_fragment_11, matrix_fragment_38,
                             matrix_fragment_39);

    condition_0 = (matrix_fragment_13 | packed_half_158 | packed_half_4) != 0;

    matrix_fragment_28 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_28),
                                       select_half_operand<2, false, false>(0u));

    multiply_accumulate_half(matrix_fragment_44, matrix_fragment_45, matrix_fragment_32,
                             matrix_fragment_33, matrix_fragment_34, matrix_fragment_35,
                             matrix_fragment_8, matrix_fragment_9, matrix_fragment_44,
                             matrix_fragment_45);

    matrix_fragment_8 = uniform_index_word_9;

    packed_half_5 = matrix_fragment_8 * 0x100u + packed_half_5;

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        int64_t(int32_t(packed_half_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(matrix_accumulator_2, matrix_accumulator_3));
    if (uniform_condition_1) {
        packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_37),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_39),
                                          select_half_operand<2, false, false>(0u));

        packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_36),
                                      select_half_operand<2, false, false>(0u));

        matrix_fragment_9 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_38),
                                          select_half_operand<2, false, false>(0u));

        matrix_fragment_43 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_43),
                                           select_half_operand<2, false, false>(packed_half_4));

        matrix_fragment_11 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_45),
                                           select_half_operand<2, false, false>(matrix_fragment_8));

        matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_42),
                                          select_half_operand<2, false, false>(packed_half_5));

        split_address_words(packed_half_4, packed_half_5,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_10 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_44),
                                           select_half_operand<2, false, false>(matrix_fragment_9));

        matrix_fragment_9 = matrix_fragment_43;

        store_four_words_streaming<CheckBounds>(
            combine_words(packed_half_4, packed_half_5), matrix_fragment_8, matrix_fragment_9,
            matrix_fragment_10, matrix_fragment_11, memory_ranges, error_counts);
    }
    uniform_condition_0 = uniform_condition_0 & uniform_condition_2;

    matrix_fragment_8 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_7),
                                      select_half_operand<2, false, false>(0u));

    matrix_fragment_29 = pack_fp8_pair(select_half_operand<2, false, false>(matrix_fragment_29),
                                       select_half_operand<2, false, false>(0u));

    packed_half_5 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_24),
                                  select_half_operand<2, false, false>(0u));

    packed_half_4 = pack_fp8_pair(select_half_operand<2, false, false>(address_word_25),
                                  select_half_operand<2, false, false>(0u));

    packed_half_6 = pack_fp8_pair(select_half_operand<2, false, false>(packed_half_6),
                                  select_half_operand<2, false, false>(0u));

    matrix_accumulator_7 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_26),
                      select_half_operand<2, false, false>(0u));

    matrix_accumulator_27 =
        pack_fp8_pair(select_half_operand<2, false, false>(matrix_accumulator_27),
                      select_half_operand<2, false, false>(0u));
    if (uniform_condition_0) {
        matrix_fragment_8 = permute_signed(matrix_fragment_8, matrix_accumulator_27, 0x5410u);

        split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                            int64_t(int32_t(address_word_0)) * int64_t(int32_t(0x10u)) +
                                combine_words(matrix_accumulator_2, matrix_accumulator_3));

        matrix_fragment_29 = permute_signed(matrix_fragment_29, packed_half_4, 0x5410u);

        packed_half_6 = permute_signed(packed_half_6, matrix_accumulator_7, 0x5410u);

        packed_half_4 = permute_signed(matrix_fragment_28, packed_half_5, 0x5410u);

        packed_half_5 = matrix_fragment_29;

        matrix_accumulator_7 = matrix_fragment_8;

        store_four_words_streaming<CheckBounds>(
            combine_words(matrix_accumulator_2, matrix_accumulator_3) + 0x400ull, packed_half_4,
            packed_half_5, packed_half_6, matrix_accumulator_7, memory_ranges, error_counts);
    }
    __syncthreads();

    __syncthreads();

    if (condition_0) {
        return;
    }

    uniform_index_word_5 = blockIdx.x;

    uniform_index_word_6 = parameters[parameter_publication_optional_low_16];
    uniform_index_word_7 = parameters[parameter_publication_optional_high_17];

    uniform_index_word_5 = uniform_index_word_5 + uniform_address_word_4;

    split_address_words(uniform_index_word_6, uniform_index_word_7,
                        int64_t(int32_t(uniform_index_word_5)) * int64_t(int32_t(0x4u)) +
                            combine_words(uniform_index_word_6, uniform_index_word_7));

    split_address_words(matrix_accumulator_2, matrix_accumulator_3,
                        combine_words(uniform_index_word_6, uniform_index_word_7));

    __threadfence();

    pipeline_write<CheckBounds>(combine_words(matrix_accumulator_2, matrix_accumulator_3), 0u,
                                memory_ranges, error_counts);
    return;
}
extern "C" __global__ __launch_bounds__(64) void phoenix_window_attention_channels_64_upsample(
    KernelArguments parameters) {
    run_window_attention_channels_64_upsample<false>(parameters.words, nullptr, nullptr);
}
